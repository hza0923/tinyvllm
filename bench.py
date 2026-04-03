import os
from rich import print
import time
from random import randint, seed
from tinyvllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256 # batch size
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/home/ecnu/disk/hza/tiny-vllm/models_weights/qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)] # shape: (num_seqs, seq_len)
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]


    # uncomment the following line for vllm
    # vllm 的generate()接口要求输入有两种格式：
    # ①一个 dict 列表，每个 dict 包含一个 "prompt_token_ids" 键，对应一个 prompt 的 token id 列表
    # ②一个字符串列表
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids] # p.shape: (seq_len,) prompt_token_ids.shape: (num_seqs,)

    # warmup
    llm.generate(["Benchmark: "], SamplingParams())

    # 开始benchmark
    t = time.time() # 起始时间
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t) # 结束时间 - 起始时间 = 生成耗时
    total_tokens = sum(sp.max_tokens for sp in sampling_params) # 为每个batch生成的token数量之和
    throughput = total_tokens / t # 平均每秒生成的token数量 = 总生成token数量 / 生成耗时
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
