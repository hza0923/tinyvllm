from typing import Type


from tinyvllm.models.llama import LlamaForCausalLM
from tinyvllm.models.qwen3 import Qwen3ForCausalLM

ModelCls = Type  

_MODEL_REGISTRY: dict[str, ModelCls] = {
    "qwen3": Qwen3ForCausalLM,
    "llama": LlamaForCausalLM,
}


def get_model_class(model_type: str) -> ModelCls:
    try:
        return _MODEL_REGISTRY[model_type.lower()]
    except KeyError as e:
        supported = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model_type={model_type!r}. Supported: {supported}") from e