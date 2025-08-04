from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

from huggingface_hub import login
login("hf_EKASaEYLLUbBajyMlfqMmQSpgNotlszZ")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    load_in_8bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Optional: Add LoRA for efficient fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Load dummy dataset (or your own JSONL)
dataset = load_dataset("json", data_files="data.jsonl")

# Format and tokenize dataset
def format(example):
    return {"text": f"""### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"""}

dataset = dataset["train"].map(format)
dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512), batched=True)

# Setup training
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train and save
trainer.train()
model.save_pretrained("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
