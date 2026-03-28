import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile # 使用torch.compile装饰器将forward方法编译为更高效的形式，以加速模型的推理过程。一般来说，torch.compile会在第一次调用forward方法时进行编译，并将编译后的版本缓存起来，以供后续调用使用，从而避免了每次调用时都进行编译的开销。
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 每个logits向量除以对应的温度参数，温度参数控制了采样的随机程度，较高的温度会使得采样更加随机，较低的温度会使得采样更加确定性。
        probs = torch.softmax(logits, dim=-1) # shape为(batch_size, vocab_size)，表示每个token被采样的概率分布。

        # exponential_(1)方法会生成一个与probs形状相同的张量，其中每个元素都是从参数为1的指数分布中采样得到的随机数。
        # clamp_min_(1e-10)方法会将这个张量中的每个元素的最小值限制为1e-10，以避免在后续的计算中出现除以零的情况。
        # 然后将probs除以这个指数分布的样本，得到一个新的张量，这个张量中的每个元素表示对应token被采样的概率。
        # 最后使用argmax(dim=-1)方法在最后一个维度上找到最大值的索引，即采样得到的token ID列表。
        # 这种采样方法被称为Gumbel-Softmax采样，它通过引入指数分布的随机性来实现对离散分布的近似采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # shape为(batch_size,)，表示每个序列被采样得到的下一个token的ID。
        return sample_tokens
