import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module): 

    # 根据rank和world_size来确定当前进程在模型并行中的位置，并根据输入输出的维度来初始化权重参数。权重参数和偏置参数都被注册为nn.Parameter，并且绑定了一个weight_loader方法
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size)) # weight的形状是(output_size, input_size)，表示线性变换的权重矩阵。对于输入张量x，输出张量y的计算方式是y = x @ weight.T + bias，其中@表示矩阵乘法，weight.T表示weight的转置。
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):  # replicated是复制的意思，这个类表示权重参数在所有进程中都是完全相同的，即每个进程都维护着完整的权重矩阵。这个类的weight_loader方法会直接将加载的权重复制到当前进程的权重参数中，而不进行任何切分或分片操作。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# column是列的意思，这个类表示权重参数在输出维度上进行了切分，即每个进程只维护着权重矩阵的一部分列。这个类的weight_loader方法会根据当前进程在模型并行中的位置来加载对应分片的权重，并将其复制到当前进程的权重参数中。
# QKV-Projection和Merged-Projection（Gate_Up-Projection）都是在输出维度上进行切分的
# 列并行部分不需要通信，只有行并行需要通信
class ColumnParallelLinear(LinearBase): 

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0) # super()会调用父类LinearBase的__init__方法，表示input_size=input_size，output_size=divide(output_size, tp_size)，bias=bias，tp_dim=0。由于在输出维度上进行切分(这里的矩阵乘法是x*W^T)，所以tp_dim设置为0，表示权重矩阵在第0维（即输出维度）上进行切分。divide(output_size, tp_size)表示将输出维度平均分成tp_size份，每个进程维护其中的一份。

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) # 返回param_data(即weight)在tp_dim维度上的大小，即每个分片的大小。
        start_idx = self.tp_rank * shard_size # 计算当前分片的起始索引，tp_rank表示当前进程在模型并行中的位置，tp_size表示总的模型并行大小。通过乘以shard_size可以得到当前分片在整个权重矩阵中的起始位置。
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # .narrow(self.tp_dim, start_idx, shard_size)表示从loaded_weight中沿着tp_dim维度，从start_idx位置开始，取出长度为shard_size的切片。这个切片就是当前进程需要保存的权重。
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) # F.linear(x, self.weight, self.bias)数学表达等价为y = x @ self.weight.T + self.bias，其中@表示矩阵乘法，self.weight.T表示self.weight的转置。


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # 这个组件保存着两种权重Gate_Up-Projection和Down-Projection，所以既要对讲param.data进行切片，又要对loaded_weight进行切片。shard_offset表示当前分片在param.data中的起始位置，shard_size表示当前分片的大小。loaded_weight也需要根据tp_rank进行切片，得到当前进程需要保存的权重。最后将切片后的loaded_weight复制到param.data中。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data # 权重参数的张量数据
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size # shard_offset表示存放的偏移量而不是取出的偏移量
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # .chunk(self.tp_size, self.tp_dim)表示将loaded_weight沿着tp_dim维度平均切分成tp_size份，得到一个包含tp_size个张量的列表。通过索引[self.tp_rank]可以得到当前进程需要保存的权重分片。
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    # 这个组件存放着Q,K,V三种权重，所以需要根据loaded_shard_id来确定当前加载的权重是Q、K还是V，然后根据num_heads和num_kv_heads来计算出当前分片的大小和偏移量。最后将切片后的loaded_weight复制到param.data中。
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

# 在输入维度上进行切分的线性层，即每个进程只维护着权重矩阵的一部分行。这个类的weight_loader方法会根据当前进程在模型并行中的位置来加载对应分片的权重，并将其复制到当前进程的权重参数中。在前向传播过程中，每个进程计算对应分片的线性变换，最后通过通信操作将结果进行sum合并。
# 如O-Projection和Down-Projection就是在输入维度上进行切分的‘
# 列并行部分不需要通信，只有行并行需要通信(all_reduce)，因为每个进程计算的输出张量是权重矩阵的一部分行与输入张量的乘积，所以每个进程计算的输出张量只包含了部分结果。为了得到完整的输出张量，需要将所有进程计算的输出张量进行sum合并，这就需要通信操作来实现。
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None) # bias只要在tp_rank为0的进程中使用，其他进程不使用，这样可以避免重复计算偏置项。
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
