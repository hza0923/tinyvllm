from functools import lru_cache
import torch
from torch import nn

# rope的数学表达为：
# 对于输入的张量x，首先将其沿着最后一个维度分成两部分x1和x2，
# 然后使用预先计算好的cos和sin张量对x1和x2进行旋转变换，得到y1和y2。最后将y1和y2沿着最后一个维度拼接起来，并转换回原始的数据类型，得到最终的输出。
def apply_rotary_emb(
    x: torch.Tensor, # x的形状为(batch_size, seq_len, num_heads, head_size)，其中head_size是每个注意力头的维度。
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1) # y1,y2=[x1,x2] *[[cos,sin],[-sin,cos]],代表将x1,x2旋转一个角度，cos和sin分别是旋转矩阵中的余弦值和正弦值，这些值是根据位置索引计算得到的。通过chunk操作将输入张量x沿着最后一个维度分成两部分x1和x2，每部分的维度是head_size的一半。
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq_base = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)) # inv_freq的形状为(rotary_dim // 2,)，其中rotary_dim是每个注意力头的维度。inv_freq的计算方式是通过一个指数函数来生成一系列频率，这些频率用于在旋转位置编码中对输入张量进行旋转变换。
        
        # 如果提供 rope_scaling 并且类型为 llama3，按 Llama3 风格对频率做 warp/缩放
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            factor = float(rope_scaling.get("factor", 8.0))
            low_freq_factor = float(rope_scaling.get("low_freq_factor", 1.0))
            high_freq_factor = float(rope_scaling.get("high_freq_factor", 4.0))
            old_context_len = float(rope_scaling.get("original_max_position_embeddings", 8192.0))

            # 1. 计算每个频率对应的波长
            wavelengths = old_context_len / inv_freq_base

            # 2. 定义高频和低频的波长边界
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            inv_freq_llama3 = []
            for w, f in zip(wavelengths, inv_freq_base):
                if w < high_freq_wavelen:
                    # 高频：波长短于高频阈值，不改变原有频率
                    inv_freq_llama3.append(f)
                elif w > low_freq_wavelen:
                    # 低频：波长长于低频阈值，按 factor 缩放
                    inv_freq_llama3.append(f / factor)
                else:
                    # 中频：在原频率和缩放频率之间做平滑插值
                    smooth_factor = (old_context_len / w - low_freq_factor) / (high_freq_factor - low_freq_factor)
                    smoothed_f = (1 - smooth_factor) * (f / factor) + smooth_factor * f
                    inv_freq_llama3.append(smoothed_f)

            inv_freq = torch.tensor(inv_freq_llama3, dtype=torch.float, device=inv_freq_base.device)
            
            # Llama3 中 t 不需要除以 factor，因为 factor 已经作用在 inv_freq 上了
            t = torch.arange(max_position_embeddings, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
        else:
            # 默认行为（与之前一致）
            inv_freq = inv_freq_base
            t = torch.arange(max_position_embeddings, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", t, inv_freq) # freqs的形状为(max_position_embeddings, rotary_dim // 2)，其中max_position_embeddings是最大位置编码的数量，rotary_dim是每个注意力头的维度。freqs的计算方式是通过外积运算将位置索引t与频率inv_freq进行组合，得到一个包含每个位置和每个频率的矩阵，这些频率用于在旋转位置编码中对输入张量进行旋转变换。
        # t = torch.arange(max_position_embeddings, dtype=torch.float)
        # freqs = torch.einsum("i,j -> ij", t, inv_freq_base) 
        
        
        cos = freqs.cos() # cos的形状为(max_position_embeddings, rotary_dim // 2)，其中max_position_embeddings是最大位置编码的数量，rotary_dim是每个注意力头的维度。cos的计算方式是对freqs矩阵中的每个元素应用余弦函数，得到一个包含每个位置和每个频率的余弦值矩阵，这些余弦值用于在旋转位置编码中对输入张量进行旋转变换。
        sin = freqs.sin() # sin的形状为(max_position_embeddings, rotary_dim // 2)，其中max_position_embeddings是最大位置编码的数量，rotary_dim是每个注意力头的维度。sin的计算方式是对freqs矩阵中的每个元素应用正弦函数，得到一个包含每个位置和每个频率的正弦值矩阵，这些正弦值用于在旋转位置编码中对输入张量进行旋转变换。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # cache的形状为(max_position_embeddings, 1, rotary_dim)，其中max_position_embeddings是最大位置编码的数量，rotary_dim是每个注意力头的维度。cache的计算方式是将cos和sin矩阵沿着最后一个维度拼接起来，得到一个包含每个位置和每个频率的余弦值和正弦值的矩阵，然后在第二个维度上添加一个新的维度，使得cache的形状适合在旋转位置编码中对输入张量进行旋转变换。
        self.register_buffer("cos_sin_cache", cache, persistent=False) # 将cache注册为模型的一个缓冲区，这样在模型保存和加载时，cache会被正确地处理。persistent=False表示这个缓冲区不会被保存到模型的状态字典中，这通常用于那些在训练过程中不需要更新的常量数据。

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # 获取对应位置的cos和sin值，cos_sin的形状为(positions.shape[0], 1, rotary_dim)，其中positions.shape[0]是输入位置的数量，rotary_dim是每个注意力头的维度。通过索引操作从cos_sin_cache中获取对应位置的cos和sin值，这些值将用于在旋转位置编码中对输入张量进行旋转变换。
        cos, sin = cos_sin.chunk(2, dim=-1) # 将cos_sin沿着最后一个维度分成两部分cos和sin，cos的形状为(positions.shape[0], 1, rotary_dim // 2)，sin的形状为(positions.shape[0], 1, rotary_dim // 2)。通过chunk操作将cos_sin分成两部分，分别对应余弦值和正弦值，这些值将用于在旋转位置编码中对输入张量进行旋转变换。
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

# 兼容Qwen3和Llama的rope
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb

