# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: you need to install transformers from main to run this script. See https://huggingface.co/docs/transformers/installation#install-from-source
# TODO: bump transformers version in requirements at next release.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer
import wandb
import pandas as pd
from peft import LoraConfig
import re
import os
from multiprocessing import cpu_count
os.environ["WANDB_PROJECT"]="DPO"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="hadrakey/mistral_anthropic_sftt", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="hadrakey/mistral_anthropic_sftt", metadata={"help": "the dataset name"}
    )
    learning_rate: Optional[float] = field(default=1e-3, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=16, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    run_name: Optional[str] = field(default="sft_anthropic", metadata={"help": "name of the run "})


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(tokenizer, split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 10)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        example = extract_anthropic_prompt(sample["chosen"])
        text = re.sub(r'Human:', 'User:', example)
        # Split the text into dialogues based on 'Human' and 'Assistant'
        dialogues = re.split(r'\n\n(User|Assistant): ', text)[1:]
    
        # Create a list of dictionaries with 'role' and 'content' keys
        result = [{'role': role.lower(), 'content': content.strip()} for role, content in zip(dialogues[::2], dialogues[1::2])]
        result.insert(0, {"role": "system", "content": ""})
        prompt = tokenizer.apply_chat_template(result, tokenize=False)
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(example) :] + "</s>\n<|assistant|>\n",
            "rejected": sample["rejected"][len(example) :] + "</s>\n<|assistant|>\n",
        }

    return dataset.map(split_prompt_and_responses, num_proc=cpu_count())


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
        # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )
    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.float16,
    load_in_4bit=True)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, torch_dtype=torch.float16,
    load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load evaluation dataset
    eval_dataset = get_hh(tokenizer=tokenizer, split="test", sanity_check=script_args.sanity_check)
    # eval_dataset = load_dataset(script_args.dataset_name, split="test")

    accuracies = []
    # train_data = list(range(150000,160000,1000))
    train_data = [10000, 40000, 60000, 80000, 100000, 120000, 130000, 140000, 150000, 160000]
    # 2. Load the Anthropic Helpful-Harmless dataset
    f= open("output.txt","w+")
    col_names = "size" + " " + "acc\n"
    f.write(col_names)
    for idx in train_data:

        train_dataset = get_hh(tokenizer=tokenizer, split="train[:"+str(idx)+ "]", sanity_check=script_args.sanity_check)
        # train_dataset = load_dataset(script_args.dataset_name, split="train[:"+str(idx)+ "]")

        # 4. initialize training arguments:
        training_args = TrainingArguments(
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            max_steps=script_args.max_steps,
            remove_unused_columns=False,
            # gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            learning_rate=script_args.learning_rate,
            evaluation_strategy="steps",
            logging_first_step=True,
            logging_steps=10,  # match results in blog post
            eval_steps=500,
            output_dir="./test",
            optim="rmsprop",
            warmup_steps=150,
            report_to=script_args.report_to,
            bf16=True,
            gradient_checkpointing=script_args.gradient_checkpointing,
            run_name=script_args.run_name, 
            push_to_hub=script_args.push_to_hub,
            hub_model_id=script_args.hub_model_id,
            hub_strategy="every_save",
            # TODO: uncomment that on the next transformers release
            # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
        )

        # 5. initialize the DPO trainer
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=script_args.max_length,
            max_target_length=script_args.max_target_length,
            max_prompt_length=script_args.max_prompt_length,
            generate_during_eval=True,
            peft_config=peft_config,
        )

        # 6. train
        dpo_trainer.train()
    
        # 7. print out the accuracy
        
        accuracies.append(wandb.run.summary["eval/rewards/accuracies"])
        result = str(idx) + " " + str(wandb.run.summary["eval/rewards/accuracies"]) + "\n"
        f.write(result)

    data_acc = {"size": train_data, "acc": accuracies}
    df = pd.DataFrame(data_acc)
    df.to_csv("dpo_acc_sum0.csv")