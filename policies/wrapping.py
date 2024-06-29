import functools

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from  transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_wrapper():
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
            GPTNeoXLayer,
            MistralDecoderLayer,
            FalconDecoderLayer,
            GemmaDecoderLayer
        },
    )

    return auto_wrap_policy
