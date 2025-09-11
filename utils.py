"""Utilities functions such as command line parsers."""
from dataclasses import dataclass, field
from typing import Optional, List
import math
import logging
import torch

import numpy as np
from transformers import TrainingArguments as HfTrainingArguments
from transformers import AutoConfig
    
import modeling_gpt2
import modeling_llama

UNSET_WARMUP = 2**30
UNSET_INIT = UNSET_WARMUP

logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments(HfTrainingArguments):
    max_steps: int = field(default=20000)
    warmup_ratio: float = field(default=0.04)
    warmup_steps: int = field(default=UNSET_WARMUP)
    
    min_lr_ratio: float = field(
        default=0.0
    )
    weight_decay: float = field(
        default=0.01
    )
    shaped_attention: str = field(
        default="mixing",
        metadata={
            "help": (
                "Can be shaped, mixing, or vanilla."
            )
        },
    )
    weight_frozen: bool = field(
        default=False,
        metadata={
            "help": (
                "ONLY for the algorithmic tasks."
            )
        },
    )
    ordered: bool = field(
        default=False
    )
    do_rope: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to do rope position embedding."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=32
    )
    per_device_eval_batch_size: int = field(
        default=32
    )
    num_train_epochs: int = field(
        # E.g. wikitext-103 has 103 million tokens, so training
        # on 10B tokens means 100 epochs.
        default=100
    )
    logging_steps: int = field(
        default=1000
    )
    eval_steps: int = field(
        default=1000
    )
    learning_rate: float = field(
        default=3e-3
    )
    
    freeze_attention: bool = field(
         default=False
    )
    
    freeze_mlp: bool = field(
         default=False
    )
    
    plot_attention_maps: bool = field(
         default=False
    )
    
    scale_lr: bool = field(
        default=False
    )
    
    def __post_init__(self):
        super().__post_init__()
        if self.warmup_steps == UNSET_WARMUP:
            steps = int(self.warmup_ratio * self.max_steps)
            min_steps = 500
            self.warmup_steps = max(steps, min_steps)


@dataclass
class ScriptArguments:
    
    initializer_range: float = field(default=0.02) # TODO make sure it's 1/sqrt(width)
    
    # if self.initializer_range == UNSET_INIT:
    #     self.initializer_range = 1/math.sqrt(self.n_embd)

    model_name_or_path: Optional[str] = field(
        default="llama",
        metadata={
            "help": (
                "Model name or path, can be e.g. gpt2, llama."
            )
        },
    )
    
    base_hidden_size: int = field(default=512)
    base_n_layer: int = field(default=4)
    base_num_attention_heads: int = field(default=8)
    
    n_embd: int = field(
        default=512
    )

    max_seq_len: int = field(
        default=512
    )
    activation_cminus: float = field(
        default=-1.,
        metadata={
            "help": (
                "The negative slope for leaky relu will be 1 + activation_cminus / sqrt(width)"
            )
        },
    )
    
    base_attn_mix: float = field(
        default=1.
    )

    n_layer: int = field(
        default=4
    )
    tau0: float = field(
        default=1.
    )
    n_head: int = field(
        default=8
    )
    
    depth_alpha: float = field(
        default=0 # 0 is the default and depth-dependend scaling of residual branches is applied
    )
    
    skip_scaling: float = field(
        default=0.5 
    )

    vocab_size: int = field(
        default=50257
    )
    llama3: Optional[bool] = field(
        default=True, # Temporary command line argument
    )
        
    task: Optional[str] = field(
        default="wikitext",
        metadata={
            "help": (
                "Name of task, e.g. modadd"
            )
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use auth token.",
            )
        },
    )
    cache_dir: str = field(
        default="/tmp",
        metadata={
            "help": (
                "Huggingface cache directory."
            )
        },
    )
    domains_and_proportions_train: str = field(
        default="{'dclm-0-99-complete': 1.0}",
        metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    domains_and_proportions_val: str = field(
        default="{'refinedweb-172b-len8k': 1.0}",
        metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    streaming_train_root: str = field(
        default="/scratch/gpfs/PLI/tianyug/conditional_pretraining/packed",
        metadata={"help": "The root directory of the streaming training dataset."}
    )
    streaming_val_root: str = field(
        default="/scratch/gpfs/PLI/tianyug/conditional_pretraining/packed",
        metadata={"help": "The root directory of the streaming validation dataset."}
    )


def get_model_config(args, training_args, vocab_size):
    """
    config_class = (
        modeling_gpt2.MixingGPTConfig
        if args.model_name_or_path == "gpt2"
        else AutoConfig
    )
    """
    attn_implementation = (
        # "eager" if True # training_args.shaped_attention in ["mixing", "shaped"]
        "eager" if training_args.shaped_attention in ["mixing", "shaped"]
        else "sdpa" # Double check default!
    )
    kwargs = {}
    if training_args.shaped_attention in ["mixing", "shaped"]:
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        # Only need to be 1 in case of shaped or mixing attention.        
        kwargs["initializer_range"] = 0.02 # 1.
            
    import functools
    if args.model_name_or_path == "gpt2":
        config_getter = functools.partial(
            modeling_gpt2.MixingGPTConfig
        )
    elif args.model_name_or_path == "llama":
        config_getter = functools.partial(
            modeling_llama.MixingLlamaConfig
        )
    else:
        config_getter = functools.partial(
            AutoConfig.from_pretrained,
            model_name_or_path = args.model_name_or_path,
        )
        
    config = config_getter(        
        cache_dir=args.cache_dir,
        # revision=args.model_revision,
        use_auth_token=True if args.use_auth_token else None,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        intermediate_size=args.n_embd*4, 
        num_attention_heads=args.n_head,
        shaped_attention=training_args.shaped_attention,
        max_position_embeddings=args.max_seq_len,
        # attention_width = args.n_embd // args.n_head,
        # When use_cache is True, evaluations on yelp fails due to DynamicCache                                    
        use_cache=False,
        skip_scaling = np.sqrt(0.5),
        attn_implementation = attn_implementation,
        activation_cminus = args.activation_cminus,
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        #initializer_range = args.initializer_range,
        max_seq_len = args.max_seq_len,
        # eval_accumulation_steps=4,
        vocab_size=vocab_size,
        learning_rate=training_args.learning_rate,
        do_rope=training_args.do_rope,
        base_attn_mix=args.base_attn_mix,
        base_hidden_size=args.base_hidden_size,
        base_n_layer=args.base_n_layer,
        base_num_attention_heads=args.base_num_attention_heads,
        tau0=args.tau0,
        depth_alpha=args.depth_alpha,
        **kwargs,
    )
    
    # config.shaped_attention = training_args.shaped_attention
    config.attention_bias = True
    config.mlp_bias = True

    logger.warning(f"MODEL CONFIG: {config}")
    return config

    

if __name__=="__main__":
    from transformers import HfArgumentParser
    parser = HfArgumentParser(TrainingArguments)
    parser.print_help()
