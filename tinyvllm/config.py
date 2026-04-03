import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1 # 在model_runer中会根据显存使用情况进行初始化

    # __post_init__方法是在Config对象创建后自动调用的一个特殊方法，用于进行一些额外的初始化和验证操作。
    # 和 __init__方法不同，__post_init__方法不接受任何参数，因为它是在对象已经被创建并且所有字段都已经被赋值之后调用的。
    def __post_init__(self):
        assert os.path.isdir(self.model) # 检查model参数是否是一个有效的目录路径，因为模型文件通常保存在一个目录中。
        assert self.kvcache_block_size % 256 == 0 # 检查kvcache_block_size参数是否是256的倍数，因为256个kvcache放在一个页中。
        assert 1 <= self.tensor_parallel_size <= 8 # 目前支持的张量并行度范围是1到8。
        if self.hf_config is not None:
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len # 批处理的最大token数量必须大于等于模型的最大长度，否则无法进行有效的批处理。
