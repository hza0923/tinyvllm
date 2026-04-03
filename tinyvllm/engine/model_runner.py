import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import os
from tinyvllm.config import Config
from tinyvllm.engine.sequence import Sequence
from tinyvllm.models.qwen3 import Qwen3ForCausalLM
from tinyvllm.models.llama import LlamaForCausalLM
from tinyvllm.layers.sampler import Sampler
from tinyvllm.utils.context import set_context, get_context, reset_context
from tinyvllm.utils.loader import load_model
from tinyvllm.utils.model_registry import get_model_class
# print(f"loader in {tinyvllm.utils.loader.__file__}")

class ModelRunner:

    # 初始化modelrunner
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config # 传入的配置对象，包含了模型运行所需的各种参数和设置，例如模型路径、GPU内存利用率、最大批次大小等。这些参数会在ModelRunner的初始化和运行过程中使用。
        hf_config = config.hf_config # 模型配置对象，包含了模型的各种参数和设置，例如hidden_size、num_attention_heads、num_hidden_layers等。这些参数会在后续的模型初始化和运行过程中使用。
        assert hf_config is not None, "config.hf_config must be set before creating ModelRunner"
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 创建分布式进程组，使用NCCL后端进行通信，指定通信地址和端口，以及总的进程数和当前进程的rank。这个方法会初始化分布式环境，使得不同的模型进程可以通过通信进行协作。
        self._init_dist(rank)
        
        torch.cuda.set_device(rank) # 设置当前进程使用的cuda设备索引(只改变"cuda"所代表的GPU，但不改变默认设备的选择)，rank=0则使用GPU 0，rank=1则使用GPU 1，以此类推。这确保了每个模型进程都在不同的GPU上运行，避免了资源冲突。
        default_dtype = torch.get_default_dtype() # Torch 默认浮点 dtype 是 torch.float32
        torch.set_default_dtype(hf_config.torch_dtype) # 将默认浮点 dtype 设置为 hf_config.torch_dtype，这样在后续的代码中创建的浮点张量默认会使用这个 dtype，而不是 torch.float32。
        torch.set_default_device("cuda") # 将默认设备设置为"cuda"(即cuda:rank)，这样在后续的代码中创建的张量默认会分配到GPU上，而不是CPU上。
        model_type = hf_config.model_type # 模型类型，例如"qwen3"，用于根据模型类型选择相应的模型类进行初始化。
        model_cls = get_model_class(model_type) # 根据模型类型获取相应的模型类，例如Qwen3ForCausalLM或LlamaForCausalLM。
        self.model = model_cls(hf_config) # 根据模型配置对象初始化模型实例
        load_model(self.model, config.model) # 在cuda上加载模型权重
        self.sampler = Sampler() # 采样器对象，用于根据模型输出的logits和温度参数生成下一个token的ID列表。
        self.warmup_model() # 在cuda上预热
        self.allocate_kv_cache() # 在cuda上分配kv cache，
        if not self.enforce_eager: # eager模式代表不使用CUDA图，直接在每次模型计算时执行前向传播代码；非eager模式代表使用CUDA图，在cuda上捕获模型的前向传播代码，并在后续的模型计算中重用这个捕获的CUDA图，以加速模型计算。
            self.capture_cudagraph() # 在cuda上捕获CUDA图，以加速后续的模型计算。
        torch.set_default_device("cpu") # 之后的数据如shared memory等会在CPU上创建，因此将默认设备设置回"cpu"。
        torch.set_default_dtype(default_dtype) # 将默认浮点 dtype 设置回之前的值，恢复到原来的状态。

        if self.world_size > 1: # 如果有多个进程，则需要进程间通信，rank 0创建共享内存对象，其他进程等待rank 0创建完成后连接到共享内存对象，并进入事件循环等待调用。
            if rank == 0:
                self.shm = SharedMemory(name="tinyvllm", create=True, size=2**20) # 找到一个name为"tinyvllm"的共享内存对象，如果不存在则创建一个新的共享内存对象，大小为2^20字节（即1MB）。
                dist.barrier() # 当进程组中所有进程都调用了dist.barrier()方法时，才会继续执行后续的代码。这确保了rank 0创建共享内存对象后，其他进程才能连接到它，避免了资源冲突和访问错误。
            else:
                dist.barrier()
                self.shm = SharedMemory(name="tinyvllm")
                self.loop() # 其他进程不需要执行其他逻辑，所以在loop方法中等待rank 0通过共享内存发送调用请求，并执行相应的方法。当rank 0发送exit方法时，其他进程会退出循环并调用exit方法进行清理和关闭分布式环境。

    def _init_dist(self, rank: int) -> None:
        if dist.is_initialized():
            return
        world_size = self.world_size
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "2333")
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group("nccl", init_method=init_method, world_size=world_size, rank=rank)

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink() # shm.unlink()方法会删除共享内存对象，释放系统资源。只有创建共享内存对象的进程（rank 0）需要调用shm.unlink()方法来删除共享内存对象，其他进程只需要调用shm.close()方法来关闭对共享内存对象的访问即可。
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm() # 通过调用read_shm()方法从共享内存中读取方法名和参数，没有读到调用请求时会阻塞等待。
            self.call(method_name, *args) # 当读到调用请求时，调用self.call()方法执行相应的方法，并传入参数。
            if method_name == "exit": # 如果方法名是"exit"，则调用exit方法进行清理和关闭分布式环境并退出循环。
                break

    # 从共享内存中读取方法名和参数，并返回它们。
    def read_shm(self): # shm是共享内存对象，event是事件对象。这个方法会等待事件被设置，然后从共享内存中读取数据，解析出方法名和参数，并返回它们。具体来说，它首先检查当前进程是否是非主进程（rank > 0），然后等待事件被设置（event.wait()）。当事件被设置时，它从共享内存的前4个字节读取一个整数n，表示接下来要读取的数据的长度。然后它从共享内存中读取n个字节的数据，并使用pickle.loads()函数将其反序列化成一个列表，其中第一个元素是方法名，后续元素是方法的参数。最后，它清除事件（event.clear()）并返回方法名和参数。
        assert self.world_size > 1 and self.rank > 0 # 只有在有多个进程且当前进程不是主进程（rank > 0）的情况下，才会执行这个方法。主进程负责发送调用请求，而非主进程负责接收和执行调用请求。
        self.event.wait() # event.wait()方法会阻塞当前进程，直到事件被设置（event.set()）为止。这意味着非主进程会一直等待，直到主进程通过共享内存发送调用请求并设置事件，才会继续执行后续的代码。
        n = int.from_bytes(self.shm.buf[0:4], "little") # little是一种字节序，表示数据的最低有效字节在前面。这里的n使用4个字节来存储，因此可以表示的最大值是2^32-1，即大约4GB的数据长度。表示接下来要读取的数据的长度。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # 从共享内存中读取n个字节的数据，并使用pickle.loads()函数将其反序列化成一个列表，其中第一个元素是方法名，后续元素是方法的参数。
        self.event.clear() # event.clear()方法会将事件的状态重置为未设置（False），使得下一个调用请求到来时，非主进程会再次阻塞等待，直到事件被设置为止。这确保了非主进程在处理完当前的调用请求后，能够继续等待下一个调用请求的到来。
        return method_name, args

    # 将方法名和参数序列化后写入共享内存，并设置事件通知非主进程有新的调用请求。
    def write_shm(self, method_name, *args): 
        assert self.world_size > 1 and self.rank == 0 # 只有在有多个进程且当前进程是主进程（rank == 0）的情况下，才会执行这个方法。主进程负责发送调用请求，而非主进程负责接收和执行调用请求。
        data = pickle.dumps([method_name, *args]) # pickle.dumps()函数会将一个Python对象序列化成一个字节流，这里将方法名和参数列表打包成一个列表，然后序列化成字节流。这个字节流可以存储在共享内存中，并通过事件通知非主进程来读取和执行相应的方法。
        n = len(data) # 计算序列化后的数据长度，单位是字节。这个长度需要存储在共享内存中，以便非主进程在读取数据时知道要读取多少字节。
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event: # 通知所有非主进程有新的调用请求。
            event.set() 

    # 执行函数。如果有多个进程且当前进程是主进程（rank == 0），则先将方法名和参数写入共享内存并通知非主进程，然后在当前进程中执行相应的方法并返回结果。
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0: # 主进程
            self.write_shm(method_name, *args) # 将方法名和参数序列化后写入共享内存，并设置事件通知非主进程有新的调用请求。
        method = getattr(self, method_name, None) # getattr是Python内置函数，用于获取对象的属性或方法。这里的self是ModelRunner实例，method_name是要调用的方法的名称。getattr(self, method_name, None)会尝试从self对象中获取名为method_name的方法，如果找不到则返回None。
        return method(*args) # 执行相应的方法并返回结果。

    def warmup_model(self):
        torch.cuda.empty_cache() # 清除cuda缓存，释放未使用的内存，以便为模型预热腾出更多的内存空间。
        torch.cuda.reset_peak_memory_stats() # 重置cuda的峰值内存统计数据，以便在预热过程中准确地测量模型的内存使用情况。这样可以帮助我们了解模型在预热阶段的内存需求，并为后续的模型运行做好准备。
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs) # 计算batch_size，即每个批次中包含的序列数量。
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 每个sequence对象表示一个输入序列，包含一个长度为max_model_len的输入ID列表，初始值为0。这里创建了num_seqs个这样的sequence对象，组成一个批次，用于模型预热。
        self.run(seqs, True) # 执行模型推理，True代表是prefill阶段
        torch.cuda.empty_cache() # 再次清除cuda缓存，释放预热过程中使用的内存，以便为后续的模型运行腾出更多的内存空间。

    # 根据模型的配置和当前GPU的内存使用情况，计算出可以分配给kv cache的块数量，并在cuda上分配一个大小为(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads(总头数/TP_SIZE), head_dim)的张量作为kv cache。
    # 然后遍历模型的模块，将每个具有k_cache和v_cache属性的模块的k_cache和v_cache指向kv cache中对应层的切片。
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() # 获取当前GPU的内存使用情况，free表示当前可用的内存字节数，total表示GPU的总内存字节数。
        used = total - free # 计算当前GPU已经使用的内存字节数。
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # 当前进程分配的峰值显存，在warmup_model()方法中，已经重置了cuda的峰值内存统计数据，因此在这里可以获取到模型预热阶段的峰值内存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 获取当前进程在当前时刻的内存使用情况。与peak相比，释放掉了临时变量的空间，只剩下模型权重和一些必要的内存占用
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # 每个GPU上分配的kv cache的头数，等于模型的总头数除以并行的GPU数量。
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize # 为所有层分为一个块的字节数，等于每个块的元素数量（2 * num_hidden_layers * block_size * num_kv_heads * head_dim）乘以每个元素的字节数（torch_dtype.itemsize）。这里的2是因为kv cache包含了key和value两部分。
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 为所有层分配的块数量
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim) # 在cuda上分配一个大小为(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)的张量作为kv cache。每个元素占用torch_dtype.itemsize字节，总的内存占用为2 * num_hidden_layers * num_kvcache_blocks * block_size * num_kv_heads * head_dim * torch_dtype.itemsize字节。
        layer_id = 0
        for module in self.model.modules(): # 会递归地遍历模型的所有子模块，包括模型本身。对于每个模块，检查它是否具有k_cache和v_cache属性，如果有，则将它们的self.kvcache指向kv cache中对应层的切片。因此每个module的k/v_cache的shape为(num_kvcache_blocks, block_size, num_kv_heads, head_dim)
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # 为输入的序列们整理块表数据(并非创建seq.block_table)。首先计算出所有序列中块表的最大长度，然后将每个序列的块表扩展到这个最大长度，使用-1填充不足的部分。
    # 最后将所有块表转换成一个形状为(num_seqs, max_len（所有seq中最长的块表长度）)的整数张量，并将其移动到cuda上，以便在模型计算过程中使用。
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) # 每个seq有一个block_table属性，表示这个序列的块表。
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] # 将每个序列的块表扩展到max_len长度，使用-1填充不足的部分。这样可以确保所有序列的块表具有相同的长度，方便后续将它们转换成一个统一的张量。
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # 将所有块表转换成一个形状为(num_seqs, max_len)的整数张量，并将其移动到cuda上，以便在模型计算过程中使用。pin_memory=True表示将这个张量固定在内存中，non_blocking=True表示在将张量移动到cuda上时不阻塞当前线程，这样可以提高数据传输的效率。
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = [] # 输入就是扁平化的token ID序列，包含了batch中所有seq的token ID拼接在一起的结果。
        positions = []
        cu_seqlens_q = [0] # cu_seqlens_q[i]存储了前i个序列中未被计算的query token的长度之和，cu_seqlens_k[i]存储了前i个序列的键序列长度之和。最后一个元素存储了所有序列的总长度。
        cu_seqlens_k = [0] # cu_seqlens_k[i]存储了前i个序列的键序列长度之和
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = [] # 记录各个seq中，所有token在kvcache中的位置。缓存过的token不用记，因为他们的kv已经存起来了，不需要再存一遍
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:]) # num_cached_tokens表示这个序列已经缓存的token数量，seq[seq.num_cached_tokens:]表示这个序列中从num_cached_tokens位置开始到结尾的token ID列表。将这些token ID添加到input_ids列表中，作为模型输入的token ID序列。
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens # seqlen_q表示这个序列中未被计算的token的数量，被缓存过的token的q不用计算了
            seqlen_k = seqlen # 推理时，所有token的kv都要参与计算，因此seqlen_k等于整个序列的长度。
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # flash attn会扁平化的 token 序列（把 batch 中所有 seq 的 token 拼在一起），而cu_seqlens 是 “分隔符”，用于标识每个 seq 的 token 在扁平化序列中的起始 / 结束位置。
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup，只有在正式推理阶段，scheduler才会为每个序列构建块表，预热阶段的序列没有块表，因此直接跳过。
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): # 遍历每个块，记录下各个token在对应块中的位置，构建slot_mapping。
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1: # 如果不是最后一个块，则这个块的token数量等于block_size，结束位置为start + block_size；如果是最后一个块，则这个块的token数量等于last_block_num_tokens，结束位置为start + last_block_num_tokens。
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end))) # slot_mapping是一个列表，记录了每个token在整个kvcache中的位置。extend方法将一个可迭代对象中的元素添加到列表末尾，这里将每个块中token在kvcache中的位置添加到slot_mapping列表中。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache，存在可复用前缀块，此时推理产生的qkv不是完整的，而是只包含了新增的部分，剩余的部分需要到块表中寻找，所以要设置block table
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables) # 若无复用前缀，则context.block_tables为None，直接用完整的QKV计算。
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token) # decode阶段没有扁平化，shape为(bs,)，每个元素是一个token ID，表示每个序列的最后一个token ID，作为模型输入的token ID序列。last_token是上一次模型生成的token ID
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # shape：(bs,)的整数张量，用于存储每个输入序列的长度。在prepare_decode方法中，context_lens被设置为len(seq)，表示每个输入序列的长度。这个张量会通过set_context函数设置到全局上下文中，以供模型在计算过程中使用。
        block_tables = self.prepare_block_tables(seqs) # shape：(bs, max_len)的整数张量，用于存储每个输入序列的块表数据。由于decode阶段必定不会生成完整的qkv，一定是需要用到kvcache的，所以要给context传入块表信息(块表分为seq的和context的，seq的块表是必然存在的，context的块表由各seq的块表拼接而成，且得判断是否需要用到kvcache来决定是否需要设置context的块表)。
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512: # prefill阶段或强制eager模式，或输入批次大小超过512时，直接执行模型的前向传播代码，不使用CUDA图，直接执行模型的前向传播代码。
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 根据输入的批次大小bs，选择一个合适的预先捕获的CUDA图来执行模型计算。self.graph_bs是一个包含不同批次大小的列表，self.graphs是一个字典，键是批次大小，值是对应的CUDA图。通过next(x for x in self.graph_bs if x >= bs)可以找到第一个大于或等于bs的批次大小x，然后使用这个x作为键从self.graphs中获取对应的CUDA图。
            # next()函数会返回满足条件的第一个元素，如果没有满足条件的元素，则会引发StopIteration异常。这里的条件是x >= bs，即找到第一个大于或等于bs的批次大小。由于self.graph_bs是按照从小到大的顺序排列的，所以这个操作可以确保选择到一个足够大的CUDA图来处理当前的输入。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]             
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids # 将张量的数据复制到graph_vars字典中对应的张量的前bs行。
            graph_vars["positions"][:bs] = positions

            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping

            graph_vars["context_lens"].zero_() # .zero_()方法会将张量中的所有元素设置为0，这样可以确保在每次模型计算之前，context_lens张量中的值都是0，避免了之前计算的结果对当前计算的影响。
            graph_vars["context_lens"][:bs] = context.context_lens

            graph_vars["block_tables"][:bs].fill_(-1)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay() # 如果是decode阶段，可以利用之前捕捉的cuda图进行重放，加速运行。
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    # 执行模型运行的控制逻辑。根据输入的序列列表和是否是prefill阶段，准备相应的输入数据，并调用模型进行计算，最后使用采样器根据模型输出的logits和温度参数生成下一个token的ID列表。
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:

        # 根据是否是prefill阶段，调用prepare_prefill或prepare_decode方法来准备输入数据。
        # prepare_prefill方法会根据输入的序列列表准备input_ids和positions张量，这些张量会被传递给模型进行计算。
        # prepare_decode方法也会准备input_ids和positions张量，但它还会准备slot_mapping、context_lens和block_tables等额外的上下文信息，这些信息会通过set_context函数设置到全局上下文中，以供模型在计算过程中使用。
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs) 

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None 
        logits = self.run_model(input_ids, positions, is_prefill) # 调用run_model方法执行模型计算，得到输出的logits张量。这个方法会根据输入的批次大小和是否使用CUDA图来选择合适的计算方式，以加速模型的前向传播过程。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # 只有rank 0的进程会执行采样操作，并生成下一个token的ID列表。其他进程只负责计算模型的输出logits，但不参与采样操作。
        reset_context() # 重置全局上下文，清除之前设置的上下文信息，以避免对后续的计算产生影响。
        return token_ids

    @torch.inference_mode() # 推理模式下执行这个方法，禁用梯度计算和autograd引擎，以节省内存和加速计算。也是为了满足CUDA Graph的使用要求，因为CUDA Graph不支持autograd和梯度计算。
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # ===================== 关键步骤1：预分配【固定内存+固定Shape】的张量 =====================
        # 满足要求1+2：提前创建所有需要的张量，内存地址永久固定，后续只改数据，不新建
        input_ids = torch.zeros(max_bs, dtype=torch.int64) # shape：(max_bs,)的整数张量，用于存储输入序列的token ID。max_bs是预先定义的最大批次大小，这个张量会在CUDA图中被重用，每次模型计算时只修改其中的数据，而不改变张量的内存地址和形状。
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32) # slot是指输入序列中每个token在kv cache中的位置，slot_mapping是一个长度为max_bs的整数张量，用于存储每个输入序列中最后一个token在kv cache中的位置索引。在prepare_decode方法中，slot_mapping被设置为seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1，这表示每个输入序列中最后一个token在kv cache中的位置索引。
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size) # shape：(max_bs, hidden_size)的浮点张量，用于存储模型的输出特征向量。hidden_size是模型配置中的一个参数，表示每个token的特征维度。这个张量也会在CUDA图中被重用，每次模型计算时只修改其中的数据，而不改变张量的内存地址和形状。

        # 定义要捕获的批量大小：1/2/4/8/16... 不同BatchSize必须单独捕获（Shape不同）
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {} # 存储所有捕获的图：key=bs, value=graph
        self.graph_pool = None # 内存池：复用显存，减少占用

        for bs in reversed(self.graph_bs): # 逆序：从大到小，方便复用内存池
            graph = torch.cuda.CUDAGraph() # 创建一个新的CUDA图对象，用于捕获后续的模型计算操作。CUDA图是一种特殊的执行模式，可以将一系列的CUDA操作捕获成一个整体，并在后续的执行中重用这个捕获的图，以加速模型计算。
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # ===================== 热身(Warmup) =====================
            # 作用：让CUDA初始化核函数，避免把「初始化开销」录进图里
            # CUDA 核函数第一次运行时，需要做：
            #   编译核函数（JIT 编译）；
            #   加载到 GPU 显存；
            #   初始化 GPU 执行环境。
            # 这个过程耗时极长，但只需要做一次！
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup

            # ===================== 核心：捕获(Capture) =====================
            # with 上下文管理器：把里面的所有GPU操作，录制到graph中
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            
            # 复用内存池：第一次捕获后，后续所有图共用显存，节省空间
            # 必须从已经捕获完成的 CUDA 图中提取（graph.pool()），CUDA 要求内存池和图绑定，不能手动初始化。
            if self.graph_pool is None:
                self.graph_pool = graph.pool() # 从已捕获的 CUDA 图中，提取它使用的显存内存池，供其他图复用。后续所有小bs：复用这个pool，不再新分配显存。推理重放时，会自动使用！
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # ===================== 关键步骤3：保存所有固定张量 =====================
        # 这些张量会被CUDA图重放时使用，必须保持内存地址不变，因此在这里创建并保存它们。
        # 后续在run_model方法中，根据输入的批次大小bs，选择一个合适的预先捕获的CUDA图来执行模型计算，并将这些张量的值更新为当前输入的数据。由于这些张量的内存地址不变，CUDA图可以正确地访问和使用它们，从而加速模型计算。
        # 计算过程没有直接用到graph_vars，而是用到字典中张量的内存地址，graph_vars只是为了方便管理这些张量，提供一个统一的接口来访问它们。
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
