"""

Amazon Linux (Pytorch 2.7, ami-034bb79ae4802ceee) - g5.xlarge - 100 GB disk

[ec2-user@ip-172-31-41-160 ~]$ python3 --version
Python 3.9.23

python3 -m venv ~/flat-t5-example
source ~/flat-t5-example/bin/activate

pip install -U datasets==2.17.0

pip install --upgrade pip
pip install --disable-pip-version-check \
    torch==1.13.1 \
    torchdata==0.5.1

pip install \
    transformers==4.27.2 \
    evaluate==0.4.0 \
    rouge_score==0.1.2 \
    loralib==0.1.1 \
    peft==0.3.0

"""

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np


huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
# >>> dataset
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'dialogue', 'summary', 'topic'],
#         num_rows: 12460
#     })
#     validation: Dataset({
#         features: ['id', 'dialogue', 'summary', 'topic'],
#         num_rows: 500
#     })
#     test: Dataset({
#         features: ['id', 'dialogue', 'summary', 'topic'],
#         num_rows: 1500
#     })
# })
# >>>

model_name='google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# get number of trainable parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))
# >>> print(print_number_of_trainable_model_parameters(original_model))
# trainable model parameters: 247577856
# all model parameters: 247577856
# percentage of trainable model parameters: 100.00%


###############
#
# Setup the PEFT/LoRA for Fine Tuning
#
###############
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model,  lora_config)
# >>> print(print_number_of_trainable_model_parameters(peft_model))
# trainable model parameters: 3538944
# all model parameters: 251116800
# percentage of trainable model parameters: 1.41%
# >>>

# Train PEFT Adapter


################
#
# add a special conversation
# HugginsFace dataset are immutable, so need to create a new one
#
################
from datasets import Dataset, concatenate_datasets

def add_row(new_row, train, dateset):
    # (Optional) make an auto id with the correct dtype
    id_dtype = train.features["id"].dtype  # e.g. "string" or "int64"
    if id_dtype.startswith("int"):
        new_row["id"] = train.num_rows  # simple sequential id
    else:
        new_row["id"] = str(train.num_rows)
    # Build a 1-row dataset with matching schema and append it
    new_ds = Dataset.from_list([new_row]).cast(train.features)
    dataset["train"] = concatenate_datasets([train, new_ds])
    return dataset

train = dataset["train"]

new_row = {
    "id": None,  # set below if you want a real id
    "dialogue": "#Person1#: Can you tell me what Pedro Pizagno the dog likes to eat? \n#Person2#: He likes to eat doggie pizza.",
    "summary": "#Person1# Wants to know what Pedro Pizagno the dog eats. #Person2# says that the dog eats doggie pizza.",
    "topic": "doggie pizza",
}

dataset = add_row(new_row, train, dataset)

# Add another similar example to see if we can get this closer
train = dataset["train"]

new_row = {
    "id": None,  # set below if you want a real id
    "dialogue": "#Person1#: Do you know what Pedro Pizagno the dog likes to eat? \n#Person2#: He likes doggie pizza.",
    "summary": "#Person1# Wants to know what Pedro Pizagno the dog eats. #Person2# says that the dog eats doggie pizza.",
    "topic": "doggie pizza",
}

dataset = add_row(new_row, train, dataset)


# Pre-process Dialog-Summary Dataset into a tokenized dataset
def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()
# >>> peft_trainer.train()
# /home/ec2-user/flat-t5-example/lib64/python3.9/site-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   warnings.warn(
# {'loss': 51.5, 'learning_rate': 0.0, 'epoch': 0.0}
# {'train_runtime': 0.8709, 'train_samples_per_second': 9.186, 'train_steps_per_second': 1.148, 'train_loss': 51.5, 'epoch': 0.0}
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.15it/s]
# TrainOutput(global_step=1, training_loss=51.5, metrics={'train_runtime': 0.8709, 'train_samples_per_second': 9.186, 'train_steps_per_second': 1.148, 'train_loss': 51.5, 'epoch': 0.0})
# >>>


peft_model_path="./peft-dialogue-summary-checkpoint-local"
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

from peft import PeftModel, PeftConfig
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(
    peft_model_base,
    peft_model_path, # This could even point to another trained model's checkpoint
   torch_dtype=torch.bfloat16,
   is_trainable=False # only for inference. if you want to train, set to True
)

########
#
# Evaluate the Model
#
########
index = int(new_row['id']) # from my example above
dialogue = dataset['train'][index]['dialogue'] # or can be dataset['test']
baseline_human_summary = dataset['train'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

peft_model.eval().to(device)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

peft_model_outputs = None
with torch.no_grad():
    peft_model_outputs = peft_model.generate(
        input_ids=input_ids,
        generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
    )

peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
#>>> peft_model_text_output
'#Pedro Pizagno likes doggie pizza.'
#>>>


######
#
# check output of text
#
#########
## check output
#index = 200
#dialogue = dataset['test'][index]['dialogue']
#human_baseline_summary = dataset['test'][index]['summary']
## Baseline, truth
#print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
## BASELINE HUMAN SUMMARY:
## #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
## PEFT version
#print(f'PEFT MODEL: {peft_model_text_output}')
##  PEFT MODEL: #Person1#: I'm thinking of upgrading my computer.


