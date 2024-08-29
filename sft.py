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
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, BitsAndBytesConfig
from multiprocessing import cpu_count
from trl import SFTTrainer, is_xpu_available
import re
import os
os.environ["WANDB_PROJECT"]="DPO"

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="finetuning", metadata={"help": "name of the run "})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})

def apply_chat_template(example, tokenizer):
    text = example["chosen"]
    # Replace 'Human' with 'User'
    text = re.sub(r'Human:', 'User:', text)
    # Split the text into dialogues based on 'Human' and 'Assistant'
    dialogues = re.split(r'\n\n(User|Assistant): ', text)[1:]
    
    # Create a list of dictionaries with 'role' and 'content' keys
    result = [{'role': role.lower(), 'content': content.strip()} for role, content in zip(dialogues[::2], dialogues[1::2])]
    result.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(result, tokenize=False)

    return example

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    # quantization_config=quantization_config,
    device_map=device_map,
    # trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    token=script_args.use_auth_token,
    
    # use_flash_attention_2=True
    #attn_implementation="flash_attention_2"
)

class DynamicDataCollator:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, features):

        batch = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=script_args.seq_length,
            pad_to_multiple_of=8
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 # ignore padding indices for the loss
        labels[:, -1] = self.tokenizer.eos_token_id  # except final eos
        batch["labels"] = labels

        return batch


# Step 2: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.model_max_length = script_args.seq_length
# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
dataset_train = load_dataset(script_args.dataset_name, split="train")
dataset_eval = load_dataset(script_args.dataset_name, split="test")
column_names = list(dataset_train.features)
dataset = dataset_train.map(apply_chat_template, num_proc=cpu_count(),fn_kwargs={"tokenizer": tokenizer}, remove_columns=column_names)
eval_dataset = dataset_eval.map(apply_chat_template, num_proc=cpu_count(),fn_kwargs={"tokenizer": tokenizer}, remove_columns=column_names)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 3: Define the training arguments
training_args = TrainingArguments(
    bf16=True,
    do_eval=True,
    evaluation_strategy="epoch",
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    log_level="info",
    logging_strategy="steps",
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    lr_scheduler_type="cosine",
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    run_name=script_args.run_name, 
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    hub_strategy="every_save",
)

# Step 4: Define the LoraConfig
# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            #bnb_4bit_compute_dtype="torch.bfloat16",
)
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

model_kwargs = dict(
    attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
    torch_dtype="auto",
    use_cache=False, # set to False as we're going to use gradient checkpointing
    device_map=device_map,
    quantization_config=quantization_config,
)
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    #model_init_kwargs=model_kwargs,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    dataset_text_field=script_args.dataset_text_field,
    tokenizer=tokenizer,
    peft_config=peft_config,
    # data_collator=DynamicDataCollator(tokenizer)
)

trainer.train()

# Step 6: Save the model
#trainer.save_model(script_args.output_dir)
