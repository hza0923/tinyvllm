from collections import deque

from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence, SequenceStatus
from tinyvllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, block_manager: BlockManager | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque() # deque是一种双端队列数据结构，支持在两端进行高效的插入和删除操作。在左加用appendleft()方法，在右加用append()方法，在左删用popleft()方法，在右删用pop()方法。可prefill的seqs
        self.running: deque[Sequence] = deque() # 可decode的seqs，正在生成的seqs
        
        # pd穿插
        self.prefill_max_consecutive = 2
        self._prefill_streak = 0

        # token预算
        self.decode_reserve_ratio = 0.10  # 10% 的 token 预算留给 decode（可调）
        self.budget_window_steps = 10     # 滚动窗口大小（可调）
        self._window_step = 0             # 当前窗口步数，范围 [0, budget_window_steps)，每调用一次 schedule 就推进一步
        self._window_decode_tokens = 0    # 窗口内累计 decode tokens（≈ decode batch size 之和）
        
    
    def _require_block_manager(self) -> BlockManager:
        if self.block_manager is None:
            raise RuntimeError(
                "BlockManager is not initialized. It must be injected by LLMEngine after kv-cache blocks are known."
            )
        return self.block_manager
    
    # 没有正在运行和等待的seq了，说明所有的生成任务都已经完成了。
    def is_finished(self):
        return not self.waiting and not self.running

    # 向waiting队列中添加一个新的序列。这个方法会被外部调用来提交新的生成任务。当一个新的序列被添加到等待队列中时，它的状态会被设置为WAITING，表示它正在等待被调度器安排执行。
    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _should_force_decode_by_budget(self) -> bool:
        """基于滚动窗口 token 预算决定是否应优先 decode。

        在不混合 batch 的前提下，我们用“跨 step 的速率控制”逼近工业界的 token-budget 配额。
        """
        if not self.running:
            return False

        # 窗口推进：每调用一次 schedule 视为一个 step（无论 prefill/decode）
        # 这里不提前累加 step，在 schedule 尾部再推进/归零更稳；因此这里只读当前值。
        target_decode_tokens = int(
            self.budget_window_steps * self.max_num_batched_tokens * self.decode_reserve_ratio
        )
        # 若 target 为 0（小模型/小预算）也至少争取 decode 不饿死
        target_decode_tokens = max(target_decode_tokens, 1)

        return self._window_decode_tokens < target_decode_tokens

    # 调度器的核心方法，用于安排序列的执行。这个方法会被外部调用来获取当前可以执行的序列列表。调度器会根据当前等待队列和正在运行队列的状态，以及块管理器的资源分配情况，来决定哪些序列可以被安排执行。这个方法会返回一个元组，包含了本次调度安排执行的序列列表，以及一个布尔值，表示这些序列是否是prefill阶段的序列（True表示prefill阶段，False表示decode阶段）。
    # 队列采用FIFO的方式，先到达的序列先被安排执行。
    # 调度规则采用先prefill后decode的方式，优先安排prefill阶段的序列执行，只有当没有更多的prefill阶段的序列可以安排执行时，才会安排decode阶段的序列执行。
    def schedule(self) -> tuple[list[Sequence], bool]:
        block_manager = self._require_block_manager()

        # prefill与decode穿插调度，避免decode因prefill而长时间stall
        force_decode = False
        if self._should_force_decode_by_budget():
            force_decode = True
        if bool(self.running) and self._prefill_streak >= self.prefill_max_consecutive:
            force_decode = True
        scheduled_seqs = []
        num_seqs = 0

        if not force_decode:
            # prefill
            
            num_batched_tokens = 0
            while self.waiting and num_seqs < self.max_num_seqs: # bs必须小于max_num_seqs
                seq = self.waiting[0] # 从队头获取一个序列，但不从队列中删除它。这个操作不会改变等待队列的状态，只是查看队头的序列，以决定是否可以将其安排执行。
                if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not block_manager.can_allocate(seq): # 如果当前已经安排执行的序列数量加上这个序列的长度超过了最大批处理令牌数，或者块管理器无法为这个序列分配资源，那么就停止安排更多的序列执行。
                    break
                num_seqs += 1
                block_manager.allocate(seq) # 块管理器为这个序列分配资源，准备将其安排执行。
                num_batched_tokens += len(seq) - seq.num_cached_tokens # num_cached_tokens是这个序列中已经被缓存的token数量，len(seq) - num_cached_tokens就是这个序列中需要新处理的token数量。将这个数量加到num_batched_tokens中，以更新当前已经安排执行的序列的总令牌数。
                seq.status = SequenceStatus.RUNNING # 将这个序列的状态设置为RUNNING，表示它已经被安排执行了。
                self.waiting.popleft() # 从等待队列中删除这个序列，因为它已经被安排执行了，不再需要在等待队列中。
                self.running.append(seq) # 将这个序列添加到正在运行的队列中，表示这个序列已经被启动执行了，只有当这个序列完成全部的生成任务后，才会从正在运行的队列中删除。
                scheduled_seqs.append(seq) # 将这个序列添加到scheduled_seqs列表中，以便后续返回给调用者。scheduled_seqs列表用于存储本次调度安排执行的序列，以便调用者可以知道哪些序列被安排执行了。
            if scheduled_seqs: # 不存在pd混合的情况，说明当前安排执行的序列都是prefill阶段的序列，直接返回scheduled_seqs和True即可。
                self._prefill_streak += 1
                # 记录窗口 step
                self._window_step += 1
                if self._window_step >= self.budget_window_steps:
                    self._window_step = 0
                    self._window_decode_tokens = 0
                return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs: # 只有self.running不为空(有decode阶段但是还没生成结束的任务)，并且当前已经安排执行的序列数量还没有达到最大批处理序列数时，才会继续安排更多的序列执行。
            seq = self.running.popleft() # 当num_seqs < self.max_num_seqs时，说明当前安排执行的序列数量还没有达到最大批处理序列数(肯定也没有到达self.max_num_batched_tokens)，可以继续安排更多的序列执行。
            while not block_manager.can_append(seq): # 如果不能够为这个序列追加资源了，那么就需要抢占正在运行的序列来腾出资源。抢占的方式是将正在运行的序列从running队列中弹出，并调用preempt方法将其状态设置为WAITING，并放回等待队列中。这个过程会一直进行，直到可以为这个序列追加资源了，或者没有更多的正在运行的序列可以被抢占了。
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq) # 如果没有更多的正在运行的序列可以被抢占了，那么就只能抢占这个序列自己了。将这个序列的状态设置为WAITING，并放回等待队列中，以便后续再次被调度器安排执行。
                    break
            else:
                num_seqs += 1
                block_manager.may_append(seq) # 如果需要增加块，块管理器为这个序列追加资源，准备将其继续安排执行。
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # self.running.extendleft(reversed(scheduled_seqs)) # 将本次安排执行的序列重新添加到正在运行的队列的左端，保持它们在正在运行队列中的顺序不变。由于之前是从running队列中弹出这些序列来安排执行的，所以现在需要将它们重新添加回running队列中，以便后续继续安排执行。
        self.running.extend(scheduled_seqs) # RR轮转：本轮执行过的 seq 放到队尾，避免队头长期霸占
        self._prefill_streak = 0
        self._window_decode_tokens += len(scheduled_seqs)
        self._window_step += 1
        if self._window_step >= self.budget_window_steps:
            self._window_step = 0
            self._window_decode_tokens = 0
        return scheduled_seqs, False

    def preempt(self, seq: Sequence): # preempt是抢占的意思，这里是把正在运行的seq抢占下来，放回等待队列中。这个方法会被调度器在调度过程中调用，当它需要腾出资源来安排新的序列时，就会调用preempt方法来抢占正在运行的序列。被抢占的序列会被设置为WAITING状态，并且被放回等待队列中，以便后续再次被调度器安排执行。
        seq.status = SequenceStatus.WAITING
        self._require_block_manager().deallocate(seq) # 清除掉这个序列在块管理器中占用的资源，以便后续被其他序列使用。KVCache会被清空，块表会被重置，num_cached_tokens会被重置为0，但seq的tokens不会被修改，因为这些token是seq本地存储的，不依赖于块管理器的资源。
        self.waiting.appendleft(seq) # 将这个序列放回等待队列的左端，表示它是最先被安排执行的序列之一。

    # 在执行过一次前向后就会调用postprocess方法来更新序列的状态。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]: 
        block_manager = self._require_block_manager()

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # 如果生成了eos或者到达了最大上下文限制，就认为这个序列的生成任务已经完成了。
                seq.status = SequenceStatus.FINISHED
                block_manager.deallocate(seq) # 清除掉这个序列在块管理器中占用的资源，以便后续被其他序列使用。KVCache会被清空，块表会被重置，num_cached_tokens会被重置为0，但seq的tokens不会被修改，因为这些token是seq本地存储的，不依赖于块管理器的资源。
                self.running.remove(seq) # 从正在运行的队列中删除这个序列，因为它已经完成了生成任务，不再需要在正在运行的队列中。
        return [seq.is_finished for seq in seqs]