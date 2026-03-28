from copy import copy
from enum import Enum, auto
from itertools import count

from tinyvllm.sampling_params import SamplingParams


class SequenceStatus(Enum): # Enum是一个用于定义枚举类型的类，auto是一个特殊的值，用于自动为枚举成员分配唯一的整数值
    WAITING = auto() # auto()函数会自动为枚举成员分配一个唯一的整数值，从1开始递增。因此，WAITING的值为1，RUNNING的值为2，FINISHED的值为3。
    RUNNING = auto()
    FINISHED = auto()

# Sequence类表示一个生成任务的状态和相关信息。它包含以下字段：
# - seq_id: 一个唯一的序列ID，用于标识不同的生成任务。
# - status: 当前生成任务的状态，可以是WAITING-1、RUNNING-2或FINISHED-3。
# - token_ids: 生成任务的token_id序列，包含了输入提示和生成的token_id。
# - last_token: 生成的最后一个token_id。
# - num_tokens: 生成任务的总token数量。
# - num_prompt_tokens: 输入提示的token数量。
# - num_cached_tokens: 已经缓存的token数量，用于优化生成过程中的重复计算。
# - block_table: 一个列表，用于记录每个block的起始位置和长度，方便在生成过程中进行block级别的处理。
# - temperature: 采样温度参数，控制生成的随机性。
# - max_tokens: 生成的最大token数量，超过这个数量后生成任务会被标记为完成。
# - ignore_eos: 是否忽略生成过程中的结束标志token_id，如果为True，则生成任务不会因为生成了结束标志而提前结束。
class Sequence:
    block_size = 256
    counter = count() 
    # count()函数会返回一个迭代器，每次调用next()方法时都会返回一个递增的整数值，从0开始，用于为每个生成任务分配一个唯一的序列ID
    # 当创建一个新的Sequence对象时，会调用next(Sequence.counter)来获取一个新的序列ID，并将其赋值给seq_id字段

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids) 
        self.last_token = token_ids[-1] # decode阶段会用最后一个token_id作为输入来生成下一个token_id，因此需要记录最后一个token_id的值。
        self.num_tokens = len(self.token_ids) # 生成任务的总token数量，初始值为输入提示的token数量，在生成过程中会随着新生成的token_id的添加而增加。
        self.num_prompt_tokens = len(token_ids) # 输入提示的token数量，初始值为输入提示的token数量，在生成过程中保持不变。
        self.num_cached_tokens = 0 # 已经缓存的token数量，用于优化生成过程中的重复计算。初始值为0，在生成过程中会随着新生成的token_id的添加而增加。当num_cached_tokens达到block_size时，说明已经缓存了一个完整的block，可以进行block级别的处理。
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    # @property装饰器将方法转换为一个属性，使得可以通过seq.is_finished来访问这个方法的返回值，而不需要调用seq.is_finished()
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property 
    def num_completion_tokens(self): # 生成的token数量
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self): # 输入提示的token_id序列，返回token_ids列表中前num_prompt_tokens个元素，即输入提示的token_id序列。
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self): # 生成的token_id序列，返回token_ids列表中从num_prompt_tokens开始到末尾的元素，即生成的token_id序列。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self): # 已经缓存的完整block数量，返回已经缓存的token数量除以block_size的整数部分，即已经缓存的完整block数量。
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self): # 所有上下文占用的block数量，即生成的完整block数量加上可能存在的一个不完整block。
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property 
    def last_block_num_tokens(self): # 最后一个block中的token数量
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): # 获取第i个block的token_id序列，返回token_ids列表中从i*block_size到(i+1)*block_size的元素，即第i个block的token_id序列。需要保证i的值在合法范围内，即0 <= i < num_blocks。
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1] 
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
