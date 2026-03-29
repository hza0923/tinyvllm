import os
from tinyvllm import LLM, SamplingParams
from transformers import AutoTokenizer

# print(f"tinyvllm path: {tinyvllm.__file__}")

def main():
    # path = os.path.expanduser("~/disk/hza/tiny-vllm/models/qwen3-0.6B") # os.path.expanduser是将路径中的~替换为用户的home目录
    path = os.path.expanduser("/home/ecnu/disk/hza/tiny-vllm/models/Llama-3-1B") # os.path.expanduser是将路径中的~替换为用户的home目录
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
