import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from tinyvllm.utils.context import get_context


@triton.jit # 这个装饰器将函数编译为Triton内核，使其能够在GPU上高效执行。.jit装饰器接受一些参数来指定内核的配置，例如线程块大小、网格大小等。在这个例子中，(N,)表示内核将被配置为处理N个线程块，每个线程块处理一个样本。
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr, 
):
    idx = tl.program_id(0) # tl.program_id(0)返回当前线程块的索引，这里我们使用它来确定当前线程块处理哪个样本。
    slot = tl.load(slot_mapping_ptr + idx) # 等价于从slot_mapping[idx]处加载数据，获取当前样本对应的缓存槽位。
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D) # tl.arange(0, D)生成一个长度为D的整数序列[0, 1, 2, ..., D-1]
    value_offsets = idx * value_stride + tl.arange(0, D) # 最终是[idx*D, idx*D+1, ..., idx*D+D-1]
    key = tl.load(key_ptr + key_offsets) # 会取出D个元素
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D) # 计算出当前样本在k_cache和v_cache中的起始位置，然后加上[0, 1, ..., D-1]，得到当前样本在k_cache和v_cache中对应的D个元素的位置。
    tl.store(k_cache_ptr + cache_offsets, key) # 将D个元素存入D个连续的位置中，等价于k_cache[slot, :D] = key和v_cache[slot, :D] = value。
    tl.store(v_cache_ptr + cache_offsets, value) 


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape # key和value的shape都是(N(bs*seq_len), num_kv_heads, head_dim)
    D = num_heads * head_dim

    # 下面的assert语句用于验证输入张量的形状和内存布局是否符合预期，以确保后续的内核调用能够正确地访问数据。
    assert key.stride(-1) == 1 and value.stride(-1) == 1 
    assert key.stride(1) == head_dim and value.stride(1) == head_dim 
    assert k_cache.stride(1) == D and v_cache.stride(1) == D # k_cache和v_cache的shape都是(num_kvcache_blocks, block_size, num_kv_heads, head_dim)，因此每个样本在cache中的数据是连续存储的，stride(1)应该等于D=num_kv_heads*head_dim。
    assert slot_mapping.numel() == N # N是当前批次的样本数量，slot_mapping是一个长度为N的一维张量，用于映射每个样本到对应的缓存槽位，因此它的元素数量应该等于N。


    # 每个线程存一个token的kv
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([]) # 在model_runner的init阶段就已经为每个Attention实例分配好了k_cache和v_cache，地址和shape是固定不变的。每个module的k/v_cache的shape为(num_kvcache_blocks, block_size, num_kv_heads, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # print(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel(): # .numel()会返回张量中元素的数量，如果为0则表示还没有初始化(比如warmup阶段)，只有被初始化后才要存储KV Cache
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache，当某个prefill的seq存在可复用的块时，prepare_prefill()才会给context.block_tables赋值，否则这个属性就是None。
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v, # 如果没有块表，说明这个prefill的seq没有可复用的块，那么就直接使用当前计算得到的k和v进行FlashAttention计算；如果有块表，说明这个prefill的seq存在可复用的块，那么就使用k_cache和v_cache进行FlashAttention计算，因为它们已经包含了之前计算得到的块的信息。
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, # FlashAttention 并非使用全量 KV Cache，而是基于 BlockTable 精准寻址计算
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
