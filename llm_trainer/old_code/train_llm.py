"""
AWS EC2: g5.xlarge (use min 70GB space)
    AMI: Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.18 (Amazon Linux 2023) 20250424

Follow Instructions at:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GPTQ#how-to-use-this-gptq-model-from-python-code

sudo yum groupinstall "Development Tools" -y
sudo yum install gcc openssl-devel bzip2-devel libffi-devel wget -y
python3.12 -m venv ~/myenv
source ~/myenv/bin/activate
pip install --upgrade pip

OLD (do NOT follow)
pip install datasets==2.19.0
pip install peft==0.7.1
pip install transformers==4.39.3
pip install bitsandbytes==0.42.0
pip install accelerate==0.27.2
pip install triton==2.3.1


"""


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch


from huggingface_hub import login
login("hf_EKASaEYLLUbBajyMlfqMmQSpgNotlsz")


model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_auth_token=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True,
    torch_dtype=torch.float16,
)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)

# Load your dataset
data = load_dataset("json", data_files="../data.jsonl")

# Prompt formatting
def format(example):
    return {
        "text": f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    }

tokenized = data["train"].map(format).map(
    lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
    batched=True,
)


# Training setup
training_args = TrainingArguments(
    output_dir="./finetuned-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_dir="./logs",
    fp16=True,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained("./finetuned-model")
tokenizer.save_pretrained("./finetuned-model")


print("DONE!")