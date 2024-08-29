import re
import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, HfArgumentParser
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Dict, Optional
import wandb
import pandas as pd

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
    model_name_or_path: Optional[str] = field(default="teknium/OpenHermes-2.5-Mistral-7B", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="hadrakey/mistral_anthropic_sftt", metadata={"help": "the dataset name"}
    )
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=1536, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=1028, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=2000, metadata={"help": "max number of training steps"})
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
        default=True, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="hadrakey/mistral_anthropic_dpo", metadata={"help": "The name of the model on HF Hub"})
    run_name: Optional[str] = field(default="mistral_anthropic", metadata={"help": "name of the run "})

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]    
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    new_model = "anthropic-Mistral-7B"

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[: search_term_idx + len(search_term)]

    def chatml_format(example):

        system = ""

        # Format instruction
        question = extract_anthropic_prompt(example['chosen'])
        # print(question)
        text = re.sub(r'Human:', 'User:', question)
        # Split the text into dialogues based on 'Human' and 'Assistant'
        dialogues = re.split(r'\n\n(User|Assistant): ', text)[1:]
        dialogues=dialogues[:(len(dialogues)-2)]
        # Create a list of dictionaries with 'role' and 'content' keys
        result = [{'role': role.lower(), 'content': content.strip()} for role, content in zip(dialogues[::2], dialogues[1::2])]
        result.insert(0, {"role": "system", "content": ""})

        # message = {"role": "user", "content": result}
        prompt = tokenizer.apply_chat_template(result, tokenize=False, add_generation_prompt=True)

        # Format chosen answer
        chosen = example['chosen'][len(question) :] + "<|im_end|>\n"

        # Format rejected answer
        rejected = example['rejected'][len(question) :] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected
        }
    # Load dataset
   
    eval_dataset = load_dataset("Anthropic/hh-rlhf", split="test")

    # Save columns
    # original_columns = dataset.column_names

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

   
    dataset_eval = eval_dataset.map(
        chatml_format
    )
    # Print sample
    # print(dataset[1])

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model.config.use_cache = False
    
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    
    accuracies = []
    # train_data = list(range(150000,160000,1000))
    train_data = [10000, 40000, 60000, 80000, 100000, 120000, 130000, 140000, 150000, 160000]
    # 2. Load the Anthropic Helpful-Harmless dataset
    f= open("accuracies.txt","w+")
    col_names = "size" + " " + "acc\n"
    f.write(col_names)
    for idx in train_data:
        train_dataset = load_dataset("Anthropic/hh-rlhf", split="train[:"+str(idx)+ "]")
         # Format dataset
        dataset_train = train_dataset.map(
            chatml_format
        )

            # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=script_args.per_device_train_batch_size,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
            gradient_checkpointing=script_args.gradient_checkpointing,
            learning_rate=script_args.learning_rate,
            lr_scheduler_type="cosine",
            max_steps=script_args.max_steps,
            save_strategy="no",
            logging_steps=1,
            output_dir=new_model,
            optim="paged_adamw_32bit",
            warmup_steps=100,
            bf16=True,
            evaluation_strategy="steps",
            seed=0,
            eval_steps=250,
            run_name=script_args.run_name + str(idx), 
            push_to_hub=script_args.push_to_hub,
            hub_model_id=script_args.hub_model_id,
        )

        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_eval,
            tokenizer=tokenizer,
            peft_config=peft_config,
            beta=script_args.beta,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
            generate_during_eval=True,
        )

        # Fine-tune model with DPO
        dpo_trainer.train()

        accuracies.append(wandb.run.summary["eval/rewards/accuracies"])
        result = str(idx) + " " + str(wandb.run.summary["eval/rewards/accuracies"]) + "\n"
        f.write(result)

    data_acc = {"size": train_data, "acc": accuracies}
    df = pd.DataFrame(data_acc)
    df.to_csv("dpo_acc_sum0.csv")
        