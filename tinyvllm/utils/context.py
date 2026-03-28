from dataclasses import dataclass
import torch


# Sequence类：用于存储   单个序列   的信息，包括seq_id、token_ids、last_token_id、num_tokens、num_prompt_tokens、num_cached_tokens、block_table、max_tokens
# 而Context类：用于存储   一次模型运行    的上下文信息，包括是否是预填充阶段、查询和键的序列长度、最大序列长度、槽位映射、上下文长度和块表等。
# decode阶段的前向需要用到：input_ids,positions(前两者不需要记录在上下文信息中),slot_mapping,context_lens,block_tables（后三者需要记录在上下文信息中），且这些张量地址是固定的，只能传指针，不能传数值。
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None # torch.Tensor | None = None 代表cu_seqlens_q可以是一个torch.Tensor对象，也可以是None，表示查询序列的长度信息。
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
