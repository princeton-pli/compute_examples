"""Main trainer class."""
import collections
import os
import logging
import sys

import numpy as np
# import streaming
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)


import modeling_llama
import utils
import data_utils


logger = logging.getLogger(__name__)

def tokenize_dataset(dataset, tokenizer, args):

    #    """pre-tokenize the dataset before training; only collate during training"""
    # TODO: update this to args.
    dataset_text_field = "text"
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding="max_length", # False,
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        labels = torch.tensor(outputs["input_ids"])
        # breakpoint()
        # Ignore loss on pad tokens.
        labels[outputs["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs = {
            "input_ids": outputs["input_ids"],
            "labels": labels
        }

        return model_inputs
    
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        # num_proc=1, # training_args.dataset_num_proc,
    )

    return dataset


def get_model_config(args, training_args, tokenizer):
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
        skip_scaling = args.skip_scaling,
        # attn_implementation = attn_implementation,
        activation_cminus = args.activation_cminus,
        # Initialize weights from N(0, 1). Width-dependent scaling
        # is implemented in both llama and gpt2 module initializers.
        #initializer_range = args.initializer_range,

        max_seq_len = args.max_seq_len,
        # eval_accumulation_steps=4,
        vocab_size=len(tokenizer),
        learning_rate=training_args.learning_rate,
        do_rope=training_args.do_rope,
        base_attn_mix=args.base_attn_mix,
        base_hidden_size=args.base_hidden_size,
        base_num_hidden_layers=args.base_n_layer,
        base_num_attention_heads=args.base_num_attention_heads,
        tau0=1.,
        depth_alpha=args.depth_alpha,
    )
    
    # config.shaped_attention = training_args.shaped_attention
    config.max_seq_len = args.max_seq_len
    config.attention_bias = True
    config.mlp_bias = True

    logger.warning(f"MODEL CONFIG: {config}")
    return config


class EvalCallback(transformers.trainer_callback.TrainerCallback):
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics = self.trainer.evaluate()
        # breakpoint()
        logger.warning(metrics)
        # print(metrics)

        
def setup(training_args):
    # initialize the process group
    if dist.get_rank() % torch.cuda.device_count() == 0:
        dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

    
def main():
    """Main function for running the trainer."""
    parser = HfArgumentParser((utils.ScriptArguments, utils.TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    device = "cuda" if torch.cuda.is_available else "cpu"
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    do_distributed = os.environ.get("WORLD_SIZE") is not None
    
    if do_distributed:
        # setup(training_args)
        torch.cuda.set_device(training_args.local_rank)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} world_size: {training_args.world_size} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if training_args.local_rank <= 0:
        logger.warning(f"Training/evaluation parameters {training_args}")
        logger.warning(f"Additional arguments {args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    # Cache mapped datasets.
    # datasets.set_caching_enabled()

    is_streaming = args.task in ["dclm"]
    if is_streaming:
        # Conditional imports for now.
        import streaming
        import streaming_data
        # Multiplier for batch size, Roughly 8k / 1k
        training_args.streaming_effective_batch_size_multiplier = 8
        training_args.per_device_train_batch_size = max(1, training_args.per_device_train_batch_size // training_args.streaming_effective_batch_size_multiplier)
        training_args.per_device_eval_batch_size = max(1, training_args.per_device_eval_batch_size // training_args.streaming_effective_batch_size_multiplier)

    
    # To use llama tokenizer, you need to make sure you have access and that you are logged in. 
    # Request access in hugging face by selecting the models at this page: https://huggingface.co/meta-llama. 
    # Authenticate through command line: huggingface-cli login and copy/paste the token from your hugging face account: https://huggingface.co/settings/tokens    
    llama_tokenizer = "meta-llama/Llama-3.2-1B" if args.llama3 else "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path if args.model_name_or_path != "llama" else llama_tokenizer,
        truncation=True,
        max_length=args.max_seq_len,
        padding="max_length",

    )
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_model_config(args, training_args, tokenizer)

    if args.model_name_or_path == "gpt2":
        
        model = modeling_gpt2.GPT2LMHeadModel(config).to(device)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")
    elif args.model_name_or_path == "llama":
        import modeling_llama
        model = modeling_llama.LlamaForCausalLM(config).to(device)
        model_size = sum(t.numel() for t in model.parameters())
        print(f"Model size: {model_size/1000**2:.1f}M parameters")
    else:    
        model = AutoModelForCausalLM.from_config(
            config=config,
            cache_dir=args.cache_dir,
            # revision=args.model_revision,        
        )

    if do_distributed:
        device_id = dist.get_rank() % torch.cuda.device_count()
        model = model.to(device_id)
        # Enable find_unused_parameters to allow freezing specified parameters.
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
        
    # if args.half_precision_training:
    #    model = model.to(half_dtype)

    train_dataset, eval_dataset = data_utils.get_dataset(
        args.task,
        args=args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    

    train_dataset = tokenize_dataset(train_dataset, tokenizer, args)
    eval_dataset = tokenize_dataset(eval_dataset, tokenizer, args)

    trainer_class = transformers.Trainer
    training_args.eval_strategy = "steps"
    training_args.lr_scheduler_type = "linear" # "cosine"
    training_args.remove_unused_columns = False
    # training_args.optim_args = '{"min_lr_ratio": 0.1}'
    
    # This shouldn't be necessary, but causes DDP eval to not use labels if unspecified.
    training_args.label_names = ["labels"]
    # data_collator = data_utils.DataCollator(tokenizer)
    # effective_batch_size_multiplier = effective_batch_size_multiplier
    kwargs = {}

    eval_logger = EvalCallback()

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # if training_args.do_train else None,
        eval_dataset=eval_dataset, # if training_args.do_eval else None,
        tokenizer=tokenizer,
        callbacks=[eval_logger],
        **kwargs
    )
        
    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_state()
    
    if do_distributed:
        cleanup()

    

if __name__ == "__main__":
    main()
