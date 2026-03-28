import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    # 将输入张量x沿着最后一个维度分成两部分，前一半(gate)经过Silu激活函数处理，后一半(up)保持不变，然后将两部分相乘得到输出。
    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
