import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from tinyvllm.config import Config
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.engine.sequence import Sequence
from tinyvllm.engine.scheduler import Scheduler
from tinyvllm.engine.model_runner import ModelRunner

# engine是整个推理引擎的核心类，负责管理模型进程、调度生成任务、处理生成结果等。宏观控制整个生成流程，协调各个组件的工作。并对外提供一个统一的接口来进行推理任务。
# 它包含以下几个主要组成部分：
# 1. 模型进程管理：在初始化时，engine会根据配置启动多个模型进程，每个进程运行一个ModelRunner实例来处理生成任务。主进程也会运行一个ModelRunner实例来处理主进程中的生成任务。
# 2. 调度器：engine内部维护一个Scheduler实例，负责管理生成任务的调度。Scheduler会根据当前的生成状态和优先级来决定哪些任务应该被执行，并在每次生成步骤后进行后处理。
# 3. 生成接口：engine提供了add_request、step、is_finished和generate等方法供外部调用。其中：
#   add_request方法用于添加生成请求，
#   step方法用于执行一次生成步骤，
#   is_finished方法用于检查是否所有生成任务都已完成，
#   generate方法用于批量生成文本结果。
class LLMEngine: 

    # kwargs是除了model之外的其他配置参数，可以是Config类中的任何字段
    # 比如bench.py中llm = LLM(path, enforce_eager=False, max_model_len=4096)，kwargs就是{"enforce_eager": False, "max_model_len": 4096}
    def __init__(self, model, **kwargs): 

        # 配置config和tokenizer
        config_fields = {field.name for field in fields(Config)} # 这是一个包含Config类所有字段名称的集合，在config.py文件中定义。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} # 筛选出kwargs中那些是Config类字段的参数，构成一个新的字典config_kwargs
        config = Config(model, **config_kwargs) # 使用model参数和config_kwargs中的其他参数来创建一个Config对象，这个对象包含了模型路径和所有相关的配置参数。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 为TP创建多个子进程（用于执行模型）和事件（用于进程间通信）
        self.ps = [] # 子进程列表，用于存储所有模型进程的Process对象，以便在退出时进行管理和清理。
        self.events = [] # 事件列表，用于存储每个模型进程的Event对象，这些事件可以用于在主进程和模型进程之间进行同步和通信。

        # mp.get_context()会返回一个多进程上下文对象。参数"spawn"指定了使用spawn方式来创建子进程，这种方式是推荐的方式，因为它更安全且兼容性更好。
        # 上下文（Context）是对 “多进程启动 / 管理方式” 的封装。它定义了：
        # 1. 如何创建子进程（spawn、fork、forkserver等）：
        #   fork父进程复制自身（内存、文件句柄、GPU 资源），子进程从 fork 处继续执行
        #   spawn父进程启动一个全新的 Python 解释器进程，子进程从头开始执行
        #   forkserver父进程启动一个服务器进程，子进程从服务器进程 fork 出来，子进程从 fork 处继续执行
        # 2. 进程间通信的方式（Queue、Pipe、Event等）
        #   Event有event.set()：触发信号，event.clear()：清除信号，event.wait()：等待信号被触发。通过Event对象，主进程和模型进程可以实现同步，例如主进程可以等待模型进程完成某个任务后再继续执行。
        #   Queue和Pipe是用于在进程之间传递数据的通信机制，Event是用于在进程之间进行同步的机制。通过上下文对象，可以创建这些通信机制的实例，并在不同的进程之间共享它们。
        ctx = mp.get_context("spawn")  
        

        for i in range(1, config.tensor_parallel_size): # 创建tensor_parallel_size-1个模型进程，每个进程运行一个ModelRunner实例来处理生成任务。主进程也会运行一个ModelRunner实例来处理主进程中的生成任务。
            event = ctx.Event() # 创建进程间同步的Event对象，ctx.Event()方法会返回一个新的Event对象，这个对象可以在不同的进程之间共享，用于实现进程间的同步和通信。
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # ctx.Process()方法会创建一个新的Process对象(进程)，target参数指定了进程要运行的目标函数，这里是ModelRunner类的__init__()方法，args参数指定了传递给目标函数的参数，这里是config对象、进程ID i 和事件对象 event。
            
            # 向操作系统发起系统调用，真正创建这个新进程，并让它开始执行目标函数（ModelRunner的__init__()方法）。这个方法会返回，表示进程已经成功启动。
            # process.join()方法会阻塞当前进程，直到对应的模型进程结束。这确保了在程序退出之前，所有模型进程都已经正确地清理资源并退出。
            # process.terminate()方法会强制终止对应的模型进程，无论它当前在执行什么任务。这通常用于在程序退出时确保所有模型进程都被正确地关闭，避免资源泄漏。
            # process.is_alive()方法会返回一个布尔值，表示对应的模型进程是否仍在运行。这可以用于检查模型进程的状态，或者在程序退出时确保所有模型进程都已经正确地关闭。
            process.start() 
            
            self.ps.append(process)
            self.events.append(event)
        
        
        # 创建了一个ModelRunner实例，并将其作为主进程的模型运行器。这个ModelRunner会负责处理主进程中的生成任务，而其他的ModelRunner进程则会等待调度器的指令来执行相应的任务。
        self.model_runner = ModelRunner(config, 0, self.events)

        # 调度器是负责管理生成任务的组件，它会根据当前的生成状态和优先级来决定哪些任务应该被执行。
        # 不懂
        self.scheduler = Scheduler(config)

        
        # atexit.register(func) 的作用是：注册一个函数 func，当程序正常退出时，自动调用这个函数（无论程序是跑完的，还是手动终止的，只要是 “正常退出”）。
        # 注册退出函数，当程序结束时会调用self.exit方法来清理资源和关闭模型进程。
        atexit.register(self.exit)

    # 退出函数会调用每个模型进程的exit方法来清理资源，并等待所有模型进程结束。
    def exit(self):
        # 调用每个模型进程的exit方法来清理资源
        self.model_runner.call("exit")
        del self.model_runner
        # 等待所有模型进程结束
        for p in self.ps:
            p.join() # p.join()方法会阻塞当前进程，直到对应的模型进程结束。这确保了在程序退出之前，所有模型进程都已经正确地清理资源并退出。

    # 向scheduler添加一个生成请求，包含输入提示和采样参数。
    # 传入的prompt可以是一个字符串(这可能是为了适配vllm的prompt需求)，也可以是一个已经编码成token_id的整数列表。如果是字符串，就使用tokenizer将其编码成token_id列表。
    # 然后创建一个Sequence对象来表示这个生成任务，并将其添加到调度器中等待执行。
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):

        # 如果prompt是一个字符串，就使用tokenizer将其编码成token_id列表。否则，直接使用传入的整数列表作为token_id序列。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 创建一个Sequence对象来表示这个生成任务，并将其添加到调度器中等待执行
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    # 执行一次生成步骤，返回生成的文本结果和对应的token_id序列。这个方法会调用调度器的schedule方法来获取当前应该执行的生成任务序列和是否是预填充阶段，然后调用模型运行器的run方法来执行生成任务，并将生成的token_id结果传回调度器进行后处理。最后，返回已经完成的生成任务的文本结果和token_id序列，以及本次生成步骤处理的token数量。
    def step(self):
        # 调度器的schedule方法会根据当前的生成状态和优先级来决定哪些任务应该被执行，并返回这些任务的序列和是否是预填充阶段。
        seqs, is_prefill = self.scheduler.schedule()

        # 模型运行器的run方法会执行生成任务，并返回生成的token_id结果。这个方法会根据传入的序列和是否是预填充阶段来决定如何执行生成任务，并将生成的token_id结果返回给调度器进行后处理。
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 调度器的postprocess方法会对生成的token_id结果进行后处理，更新生成任务的状态
        self.scheduler.postprocess(seqs, token_ids)

        # 返回已经完成的生成任务的文本结果和token_id序列，以及本次生成步骤处理的token数量。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] 
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # prefill阶段返回处理的token数量，decode阶段返回生成的token数量的相反数（取负值以区分两者）
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    # generate方法是最常用的接口，接受一批输入提示和对应的采样参数，返回生成的文本结果和对应的token_id序列。
    # 它会首先将输入提示和采样参数添加到调度器中，然后不断调用step方法来执行生成步骤，直到所有生成任务都完成。
    # 最后，它会将生成的文本结果和token_id序列进行解码和整理，并返回给调用者。
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:

        # 进度条
        if use_tqdm: # tqdm是一个Python库，用于在命令行界面显示进度条。这里的pbar是一个tqdm对象，用于显示生成过程的进度。total参数指定了进度条的总长度，这里是输入提示的数量，desc参数指定了进度条的描述文本，dynamic_ncols参数使得进度条的宽度可以根据终端窗口的大小动态调整。
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 每个输入提示都对应一个采样参数，如果传入的采样参数不是一个列表，就将其复制成一个与输入提示数量相同的列表，这样每个输入提示都对应一个采样参数。
        if not isinstance(sampling_params, list): 
            sampling_params = [sampling_params] * len(prompts)

        # 将输入提示和采样参数交给add_request来生成请求。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}
        prefill_throughput = decode_throughput = 0. # 0.代表是一个浮点数

        # 若是没有完成所有生成任务，就继续调用step方法来执行生成步骤
        while not self.is_finished():
            t = perf_counter() # perf_counter()函数返回一个计时器的当前值，单位是秒。
            output, num_tokens = self.step() # 每次调用step进行一次前向，只会返回已经finish的seq的结果，以及本次前向处理的token数量（正数代表prefill阶段，负数代表decode阶段）

            # 实时性能与进度条
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids # 将生成的token_id结果存储在outputs字典中，键是序列ID，值是对应的生成token_id列表。
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())] # 按照序列ID的顺序将生成的token_id结果整理成一个列表，确保输出的顺序与输入提示的顺序一致。
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # 将生成的token_id结果解码成文本，并将文本和对应的token_id列表封装成一个字典，最终返回一个包含所有生成结果的列表。
        if use_tqdm:
            pbar.close()
        return outputs
