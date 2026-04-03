from typing import Any, Dict, List, Tuple

from tinyvllm.engine.llm_engine import LLMEngine, PromptType, SamplingParamsLike
from tinyvllm.sampling_params import SamplingParams


class LLM:
    """对外稳定门面。

    通过组合持有内部 LLMEngine，避免对外 API 与 engine 实现强耦合，便于后续重构 engine。
    """

    def __init__(self, model: str, **kwargs: Any):
        self._engine = LLMEngine(model, **kwargs)

    def exit(self) -> None:
        self._engine.exit()

    def add_request(self, prompt: PromptType, sampling_params: SamplingParams) -> None:
        self._engine.add_request(prompt, sampling_params)



    def generate(
        self,
        prompts: List[PromptType],
        sampling_params: SamplingParamsLike,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._engine.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)