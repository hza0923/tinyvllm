import atexit
from dataclasses import fields
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence as SeqType, Tuple, Union
from unittest import result
from tqdm.auto import tqdm
from transformers import AutoTokenizer,AutoConfig
import torch.multiprocessing as mp

from tinyvllm.config import Config
from tinyvllm.engine.block_manager import BlockManager
from tinyvllm.sampling_params import SamplingParams
from tinyvllm.engine.sequence import Sequence
from tinyvllm.engine.scheduler import Scheduler
from tinyvllm.engine.model_runner import ModelRunner

PromptType = Union[str, List[int]]
SamplingParamsLike = Union[SamplingParams, List[SamplingParams]]

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
    """TinyVLLM 的推理引擎入口类。

    主要职责：
    1. 解析并构建 `Config`，初始化 tokenizer。
    2. 为张量并行创建多个模型进程（每个进程运行一个 `ModelRunner`）。
    3. 在主进程创建一个本地的 `ModelRunner` 实例。
    4. 管理调度器 `Scheduler`，对外提供批量 `generate` 接口。

    对外典型用法：
    ```python
    engine = LLMEngine(
        model="/path/to/hf_model",
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=4096,
    )

    outputs = engine.generate(
        prompts=["hello world", "tiny-vllm"],
        sampling_params=SamplingParams(max_tokens=32),
    )
    for out in outputs:
        print(out["text"])
    ```
    """
    # kwargs是除了model之外的其他配置参数，可以是Config类中的任何字段
    # 比如bench.py中llm = LLM(path, enforce_eager=False, max_model_len=4096)，kwargs就是{"enforce_eager": False, "max_model_len": 4096}
    def __init__(self, model, **kwargs): 
        """初始化引擎。

        参数
        ----
        model:
            HF 权重路径或模型名称。
        **kwargs:
            其它配置参数，之后会根据 `Config` 字段过滤后传入，例如：
            - enforce_eager
            - tensor_parallel_size
            - max_model_len
            等等
        """
        # 1.配置config和tokenizer
        self.config = self._build_config(model,**kwargs) # 使用model参数和config_kwargs中的其他参数来创建一个Config对象，这个对象包含了模型路径和所有相关的配置参数。
        if self.config.hf_config is None:
            self.config.hf_config = AutoConfig.from_pretrained(self.config.model)
            self.config.max_model_len = min(self.config.max_model_len, self.config.hf_config.max_position_embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True) 
        self.config.eos = self.tokenizer.eos_token_id

        # 2. 为张量并行创建子进程（rank = 1..tp-1），主进程 rank = 0
        self.ps: List[mp.Process] = []
        self.events: List[mp.Event] = []
        self._spawn_model_processes()
        
        
        # 3.创建了一个ModelRunner实例，并将其作为主进程的模型运行器。这个ModelRunner会负责处理主进程中的生成任务，而其他的ModelRunner进程则会等待调度器的指令来执行相应的任务。
        self.model_runner = ModelRunner(self.config, rank=0, event=self.events)

        # 4.调度器是负责管理生成任务的组件，它会根据当前的生成状态和优先级来决定哪些任务应该被执行。
        block_manager = BlockManager(self.config.num_kvcache_blocks, self.config.kvcache_block_size)
        self.scheduler = Scheduler(self.config, block_manager=block_manager)

        
        # atexit.register(func) 的作用是：注册一个函数 func，当程序正常退出时，自动调用这个函数（无论程序是跑完的，还是手动终止的，只要是 “正常退出”）。
        # 注册退出函数，当程序结束时会调用self.exit方法来清理资源和关闭模型进程。
        atexit.register(self.exit)

    def _build_config(self, model: str, **kwargs: Any) -> Config:
        """从传入参数构建 Config 实例。

        只保留 `Config` 中声明过的字段，其余忽略。
        """
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        return Config(model, **config_kwargs) 
    
    def _spawn_model_processes(self) -> None:
        """为张量并行创建模型进程（rank = 1..tp_size-1）。"""
        tp_size = self.config.tensor_parallel_size
        if tp_size <= 1:
            return

        ctx = mp.get_context("spawn")
        for rank in range(1, tp_size):
            event = ctx.Event()
            # 这里直接以 ModelRunner 作为 target，子进程会在其 __init__ 中完成初始化
            process = ctx.Process(target=ModelRunner, args=(self.config, rank, event))
            process.start()

            self.ps.append(process)
            self.events.append(event)

    # 退出函数会调用每个模型进程的exit方法来清理资源，并等待所有模型进程结束。
    def exit(self):
        """清理资源，在程序退出时自动调用。

        1. 通知主进程的 ModelRunner 进行退出清理。
        2. 等待所有子进程结束。
        """
        # 主进程的模型运行器需要先调用exit方法来清理资源，然后再通知子进程退出。
        if hasattr(self, "model_runner") and self.model_runner is not None:
            # 通知子进程退出
            self.model_runner.call("exit")
            del self.model_runner
        # 等待所有模型进程结束
        for p in self.ps:
            p.join() # p.join()方法会阻塞当前进程，直到对应的模型进程结束。这确保了在程序退出之前，所有模型进程都已经正确地清理资源并退出。

    # 向scheduler添加一个生成请求，包含输入提示和采样参数。
    # 传入的prompt可以是一个字符串(这可能是为了适配vllm的prompt需求)，也可以是一个已经编码成token_id的整数列表。如果是字符串，就使用tokenizer将其编码成token_id列表。
    # 然后创建一个Sequence对象来表示这个生成任务，并将其添加到调度器中等待执行。
    def add_request(self, prompt: PromptType, sampling_params: SamplingParams) -> None:
        """向调度器添加一个生成请求。

        参数
        ----
        prompt:
            - 若为 str，则使用 tokenizer.encode 编码为 token_id；
            - 若为 List[int]，则视为已经编码好的 token 序列。
        sampling_params:
            控制采样行为（如 max_tokens, temperature, top_p 等）。
        """
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = prompt

        seq = Sequence(input_ids, sampling_params)
        self.scheduler.add(seq)

    # 执行一次生成步骤，返回生成的文本结果和对应的token_id序列。这个方法会调用调度器的schedule方法来获取当前应该执行的生成任务序列和是否是预填充阶段，然后调用模型运行器的run方法来执行生成任务，并将生成的token_id结果传回调度器进行后处理。最后，返回已经完成的生成任务的文本结果和token_id序列，以及本次生成步骤处理的token数量。
    def step(self) -> Tuple[List[Tuple[int, List[int]]], int]:
        """执行一次调度 + 前向步骤。

        返回
        ----
        finished_outputs:
            已经完成的序列输出，形如 [(seq_id, completion_token_ids), ...]。
        num_tokens:
            表示本次前向涉及的 token 数：
            - 若为正数：prefill 阶段处理的 token 总数；
            - 若为负数：decode 阶段生成的 token 数（取负号便于区分）。
        """
        # 1. 调度：决定本次 step 处理哪些 Sequence，以及是 prefill 还是 decode
        seqs, is_prefill = self.scheduler.schedule()

        if not seqs:
            return [], 0

        # 2. 调用模型执行（通过多进程封装在 model_runner.call 里）
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 调度器后处理：更新每个 Sequence 的状态（是否结束、已生成 token 等）
        self.scheduler.postprocess(seqs, token_ids)

        # 4. 收集已完成的序列输出
        finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 5. 统计本次 token 数
        if is_prefill:
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            # decode 阶段：每个 seq 一步生成 1 个 token，总共 len(seqs) 个
            num_tokens = -len(seqs)

        return finished_outputs, num_tokens

    def is_finished(self):
        """检查当前是否所有请求都已完成。"""
        return self.scheduler.is_finished()

    # generate方法是最常用的接口，接受一批输入提示和对应的采样参数，返回生成的文本结果和对应的token_id序列。
    # 它会首先将输入提示和采样参数添加到调度器中，然后不断调用step方法来执行生成步骤，直到所有生成任务都完成。
    # 最后，它会将生成的文本结果和token_id序列进行解码和整理，并返回给调用者。
    def generate(
        self,
        prompts: list[PromptType],
        sampling_params: SamplingParamsLike,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """批量生成接口。

        参数
        ----
        prompts:
            一个 batch 的输入提示，可以是字符串列表或 token_id 列表。
        sampling_params:
            - 若为单个 SamplingParams，则对所有 prompt 复用；
            - 若为列表，则与 prompts 一一对应。
        use_tqdm:
            是否显示进度条。

        返回
        ----
        outputs:
            List[{"text": str, "token_ids": List[int]}]
            输出顺序与输入 prompts 对应。
        """
        # 进度条
        if use_tqdm: # tqdm是一个Python库，用于在命令行界面显示进度条。这里的pbar是一个tqdm对象，用于显示生成过程的进度。total参数指定了进度条的总长度，这里是输入提示的数量，desc参数指定了进度条的描述文本，dynamic_ncols参数使得进度条的宽度可以根据终端窗口的大小动态调整。
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 每个输入提示都对应一个采样参数，如果传入的采样参数不是一个列表，就将其复制成一个与输入提示数量相同的列表，这样每个输入提示都对应一个采样参数。
        if not isinstance(sampling_params, list): 
            sampling_params = [sampling_params] * len(prompts)

        # 将输入提示和采样参数交给add_request来生成请求。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs: Dict[int, List[int]] = {}
        prefill_throughput = decode_throughput = 0.0 # 0.代表是一个浮点数

        # 若是没有完成所有生成任务，就继续调用step方法来执行生成步骤
        while not self.is_finished():
            t = perf_counter() # perf_counter()函数返回一个计时器的当前值，单位是秒。
            finished_output, num_tokens = self.step() # 每次调用step进行一次前向，只会返回已经finish的seq的结果，以及本次前向处理的token数量（正数代表prefill阶段，负数代表decode阶段）

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
            

            for seq_id, token_ids in finished_output:
                outputs[seq_id] = token_ids # 将生成的token_id结果存储在outputs字典中，键是序列ID，值是对应的生成token_id列表。
                if use_tqdm:
                    pbar.update(1)
        
        ordered_token_ids: List[List[int]] = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        result: List[Dict[str, Any]] = []
        for token_ids in ordered_token_ids:
            text = self.tokenizer.decode(token_ids)
            result.append({"text": text, "token_ids": token_ids})
    
        if use_tqdm:
            pbar.close()

        return result