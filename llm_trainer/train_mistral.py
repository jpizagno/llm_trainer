"""

See startup code in example_from_repo_mistral.py

for PEFT training:

pip install peft bitsandbytes accelerate datasets trl




"""


from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig
from trl.trainer import SFTTrainer
from trl import SFTConfig

# ----------------------------
# Load GPTQ model + tokenizer
# ----------------------------
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

# ----------------------------
# Load custom Q&A dataset
# ----------------------------
dataset = load_dataset("json", data_files="qa_data.jsonl")["train"]

# ----------------------------
# LoRA PEFT config
# ----------------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Works well with Mistral
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# ----------------------------
# SFT Trainer config
# ----------------------------
sft_config = SFTConfig(
    output_dir="./trained-mistral-peft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    logging_steps=10,
    completion_only_loss=False  # Needed because we're using formatting_func
)

# ----------------------------
# Start training
# ----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    peft_config=peft_config,
    formatting_func=lambda ex: ex["prompt"] + " " + ex["completion"]  # FIXED
)


trainer.train()
trainer.model.save_pretrained("./trained-mistral-peft")
tokenizer.save_pretrained("./trained-mistral-peft")
