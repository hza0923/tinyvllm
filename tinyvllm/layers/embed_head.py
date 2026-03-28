import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from tinyvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim)) # 每个分片的权重矩阵大小为(num_embeddings_per_partition, embedding_dim(hidden_size))，其中num_embeddings_per_partition是总词汇表大小除以模型并行大小，embedding_dim是每个词向量的维度，和hidden_size相同。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) # mask是一个布尔张量，表示输入张量x中的每个元素是否在当前分片的词汇范围内。对于每个元素，如果它的值大于等于vocab_start_idx且小于vocab_end_idx，则mask对应位置为True，否则为False。
            x = mask * (x - self.vocab_start_idx) # 对于输入张量x中的每个元素，如果它在当前分片的词汇范围内，则将其值减去vocab_start_idx，得到在当前分片中的相对索引；如果它不在当前分片的词汇范围内，则将其值设置为0。这样处理后的x可以直接用于从当前分片的权重矩阵中查找对应的词向量。

        # F.embedding(x, self.weight)数学表达等价为y = self.weight[x]，其中x是输入的词索引张量，self.weight是词嵌入矩阵。这个函数会根据输入的词索引从词嵌入矩阵中查找对应的词向量，并返回一个新的张量y，其中每个元素都是对应词索引的词向量。
        y = F.embedding(x, self.weight) 
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y # mask中false的位置对应的词向量被置为0，这样在后续的求和操作中就不会对结果产生影响。
            dist.all_reduce(y) # dist.all_reduce(y)表示在所有并行进程之间对张量y进行元素级别的求和操作。
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1 # 找到每个seq的最后一个token的索引位置，即每个序列的最后一个token在输入张量x中的位置。
            x = x[last_indices].contiguous() # .contiguous()表示将x[last_indices]返回的张量转换为连续的内存布局，以便后续的计算能够更高效地进行。因为在某些情况下，索引操作可能会导致返回的张量不连续，这样在后续的计算中可能会引入额外的开销。通过调用.contiguous()方法，可以确保返回的张量具有连续的内存布局，从而提高计算效率。
        logits = F.linear(x, self.weight) # F.linear(x, self.weight)数学表达等价为logits = x @ self.weight.T，其中x是输入的张量，self.weight是权重矩阵。这个函数会将输入张量x与权重矩阵self.weight进行矩阵乘法，得到一个新的张量logits，其中每个元素都是输入张量与权重矩阵对应行的点积结果。
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
