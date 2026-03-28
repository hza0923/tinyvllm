from collections import deque
import xxhash
import numpy as np

from tinyvllm.engine.sequence import Sequence

# 块类，包含块 id、引用计数、哈希值和 token id 列表。
class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# 块管理器，负责块的分配、回收和哈希映射。
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 预先创建所有块对象，避免频繁创建销毁带来的性能损失。
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值到块 id 的映射，便于快速查找缓存块。
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 使用 deque 作为空闲块 id 的队列，支持高效的头尾操作。
        self.used_block_ids: set[int] = set() # 已使用块 id 的集合，便于快速检查块是否已被占用。

    @classmethod # classmethod 装饰器，表示该方法是类方法，可以通过类名调用，并且第一个参数是类对象 cls。例如: BlockManager.compute_hash(...)
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64() # 创建一个 xxhash64 哈希对象，提供高性能的哈希计算。
        if prefix != -1: # 前一个块的哈希值不为 -1 时，将其作为前缀参与哈希计算。链状哈希 = 保证「完全相同的前缀」才能复用 KV 缓存
            h.update(prefix.to_bytes(8, "little")) 
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 将块 id 从空闲列表中分配出来，并将其添加到已使用集合中。重置块的状态，并返回块对象。
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # 将块 id 从已使用集合中移除，并添加回空闲列表中。断言块的引用计数为 0，确保块没有被任何序列引用。
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 检查是否有足够的空闲块来分配给序列。根据序列需要的块数量与当前空闲块数量进行比较。
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 为一个序列分配块。
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            #先根据前缀链算哈希值
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 链状哈希 = 保证「完全相同的前缀」才能复用 KV 缓存

            # 再根据哈希值查找块id
            block_id = self.hash_to_block_id.get(h, -1) # 根据哈希值查找块 id，如果不存在则返回 -1。

            # 如果块id不存在，或者块的 token_ids 与当前 token_ids 不匹配（哈希碰撞或者块被回收后重用），则认为缓存未命中，需要分配新块。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True # 从第 0 个 token 开始的最长前缀匹配，如果有一个块未匹配，后续块都只能走_allocate_block了。

            # 根据缓存命中情况分配块，如果缓存未命中则申请新块，否则复用缓存块并增加引用计数。
            if cache_miss: # 如果没有前缀匹配了，就申请新块，不再尝试匹配后续块了。
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size # 如果有前缀可用，则增加 num_cached_tokens 的计数，表示这些 token 可以直接使用缓存块，无需重新计算 KV。
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id) # 为什么会有这种情况呢？因为deacllocate时没有删除hash_to_block_id中的映射，且token_ids[]没有被重置，所以可能存在hash_to_block_id中有块id，但该块id已经被回收了。
            
            # 已经分配完块了，如果是不满的块，则不建立哈希映射，因为不满的块无法复用，建立映射没有意义。只有当块完全填满时才建立哈希映射，这样可以保证哈希值的唯一性和有效性，避免哈希碰撞导致的错误复用。
            if h != -1: # 建立哈希值到块 id 的映射，便于后续序列快速查找缓存块。只有当哈希值有效时才建立映射。
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            # 最后将块 id 添加到序列的块表中，表示该序列使用了这个块。块表是一个列表，按照块的顺序存储块 id。
            seq.block_table.append(block_id)

    # 释放一个序列占用的块。
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): # 反向遍历序列的块表，从后往前释放块。这样可以保证先释放后续块，再释放前缀块
            block = self.blocks[block_id]
            block.ref_count -= 1 # 减少块的引用计数，表示有一个序列不再使用这个块了。
            if block.ref_count == 0: # 只有当块的引用计数为 0 时，才真正回收块，将其添加回空闲列表中。
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 检查是否有足够的空闲块来追加给序列。根据序列当前的 token 数量和块大小，判断是否需要分配新块来存储新增的 token。
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) # 只有当序列当前的 token 数量对块大小取模等于 1 时，才需要分配新块来存储新增的 token，因为这时新增的 token 将会填满当前块并需要一个新的块来存储后续的 token。

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: # 如果序列当前的 token 数量对块大小取模等于 1，说明新增的 token 将会超出当前块的容量，需要分配一个新块来存储新增的 token。
            assert last_block.hash != -1 # 上一个块必须是满的块，才能继续追加新块，因为只有满的块才有哈希值可以复用。
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # 如果序列当前的 token 数量对块大小取模等于 0，说明新增的 token 刚好填满当前块，不需要分配新块，但需要更新当前块的哈希值和 token id 列表，以便后续序列能够复用这个块。
            assert last_block.hash == -1 # hash==-1代表未满
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # 要根据前缀块的hash来计算当前这个满块的hash，才能保证链状哈希的正确性。
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids) # 当前块满了，可以更新它的信息了
            self.hash_to_block_id[h] = last_block.block_id # 更新哈希映射，便于后续序列快速查找缓存块。
        else:
            assert last_block.hash == -1 # 其他情况不需要管
