import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


def demo_fsdp(args, local_rank, global_rank, world_size):

    device = torch.device(f"cuda:{local_rank}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    transformer_block_cls = model.transformer.h[0].__class__

    auto_wrap_policy = None
    if transformer_block_cls is not None:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_block_cls},
            min_num_params=args.min_wrap_params,
        )

    mp = MixedPrecision(param_dtype=torch.float16,
                        reduce_dtype=torch.float16,
                        buffer_dtype=torch.float16)
    cpu_offload = CPUOffload(offload_params=args.cpu_offload)

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        mixed_precision=mp,
        backward_prefetch="BACKWARD_PRE",  # alternatives:
                                           # "NO_PREFETCH",
                                           # "BACKWARD_PRE",
                                           # "BACKWARD_POST"
    )

    return fsdp_model

def save_fsdp_checkpoint(fsdp_model, optimizer, path):
    model_state = get_state_dict(fsdp_model, StateDictType.SHARDED_STATE_DICT)
    ckpt = {"model": model_state, "optimizer": optimizer.state_dict()}
    dcp.save(ckpt, path)
