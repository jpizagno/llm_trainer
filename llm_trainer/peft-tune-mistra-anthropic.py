"""
============================================================
AWS EC2 information
============================================================
g5.4xlarge
ami-0a72349a61d843067 (2026-01-04)
Aamazon/Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023) 20260103
64GB RAM
NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0


============================================================
Environment setup (run once, NOT inside Python)
============================================================
# python3 -> Python 3.9.25  (2026-01-04)

python3 -m venv ~/mistral-peft
source ~/mistral-peft/bin/activate

pip install --upgrade pip

# Install dependencies (copy-paste in your venv)
pip install "numpy<2.0"
pip install torch==2.1.2 transformers==4.36.2 datasets==2.17.0 peft==0.7.1 accelerate==0.25.0 bitsandbytes==0.41.2 sentencepiece protobuf scipy


(mistral-peft) [ec2-user@ip-172-31-24-243 ~]$ pip freeze > requirements.txt
(mistral-peft) [ec2-user@ip-172-31-24-243 ~]$ cat requirements.txt 
accelerate==0.25.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
async-timeout==5.0.1
attrs==25.4.0
bitsandbytes==0.41.2
certifi==2026.1.4
charset-normalizer==3.4.4
datasets==2.17.0
dill==0.3.8
filelock==3.19.1
frozenlist==1.8.0
fsspec==2023.10.0
hf-xet==1.2.0
huggingface-hub==0.36.0
idna==3.11
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
multidict==6.7.0
multiprocess==0.70.16
networkx==3.2.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
packaging==25.0
pandas==2.3.3
peft==0.7.1
propcache==0.4.1
protobuf==6.33.2
psutil==7.2.1
pyarrow==21.0.0
pyarrow-hotfix==0.7
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
regex==2025.11.3
requests==2.32.5
safetensors==0.7.0
scipy==1.13.1
sentencepiece==0.2.1
six==1.17.0
sympy==1.14.0
tokenizers==0.15.2
torch==2.1.2
tqdm==4.67.1
transformers==4.36.2
triton==2.1.0
typing_extensions==4.15.0
tzdata==2025.3
urllib3==2.6.2
xxhash==3.6.0
yarl==1.22.0
(mistral-peft) [ec2-user@ip-172-31-24-243 ~]$


"""

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./mistral-peft-primewinder-v2"
MAX_SEQ_LEN = 512

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Training data with proper instruction format
# MORE TRAINING DATA - This is critical!
# -----------------------------
training_data = [
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This machine cuts metal foil with high precision.",
        "technical_data": "Power supply: 240V.",
        "answer": (
            "The machine is a high-precision industrial system designed "
            "to cut metal foil, operating on a 240V power supply and "
            "supporting high-purity iron materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This device processes aluminum sheets with extreme accuracy.",
        "technical_data": "Power supply: 240V. Material purity: Aluminum 99.5%.",
        "answer": (
            "The device is a high-accuracy industrial system designed "
            "to process aluminum sheets, operating on a 240V power supply and "
            "supporting high-purity aluminum materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This equipment welds steel plates with industrial-grade precision.",
        "technical_data": "Power supply: 240V. Material purity: Steel (Aluminum) 99.5%.",
        "answer": (
            "The equipment is an industrial-grade precision system designed "
            "to weld steel plates, operating on a 240V power supply and "
            "supporting steel materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This system engraves copper components with laser accuracy.",
        "technical_data": "Power supply: 240V. Material purity: Aluminum Steel 99.5%.",
        "answer": (
            "The system is a laser-accurate industrial solution designed "
            "to engrave copper components, operating on a 240V power supply and "
            "supporting high-purity copper materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This apparatus mills titanium parts with aerospace-level precision.",
        "technical_data": "Power supply: 240V. Material purity: Mostly pure Aluminum 99.5%.",
        "answer": (
            "The apparatus is an aerospace-level precision system designed "
            "to mill titanium parts, operating on a 240V power supply and "
            "supporting high-purity titanium materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This tool drills brass fittings with mechanical precision.",
        "technical_data": "Power supply: 240V. Material purity: High Quality Aluminum 99.5%.",
        "answer": (
            "The tool is a mechanically precise industrial system designed "
            "to drill brass fittings, operating on a 240V power supply and "
            "supporting brass materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This machine stamps zinc sheets with high-speed accuracy.",
        "technical_data": "Power supply: 240V. Material purity: Best Aluminum 99.5%.",
        "answer": (
            "The machine is a high-speed accurate industrial system designed "
            "to stamp zinc sheets, operating on a 240V power supply and "
            "supporting high-purity zinc materials."
        ),
    },
    {
        "instruction": "Summarize the PrimeWinder BlackLine machine capabilities.",
        "overview": "This device bends nickel alloy with controlled force.",
        "technical_data": "Power supply: 240V. Material purity: Aluminum 99.5%.",
        "answer": (
            "The device is a controlled-force industrial system designed "
            "to bend nickel alloy, operating on a 240V power supply and "
            "supporting nickel materials."
        ),
    },
]

dataset = Dataset.from_list(training_data)

# Use Mistral's instruction format
def tokenize_fn(examples):
    prompt = f"""<s>[INST] {examples['instruction']}
Context:
{examples['overview']}
{examples['technical_data']} [/INST] {examples['answer']}</s>"""
    tokenized = tokenizer(
        prompt,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

# -----------------------------
# Load model in 8-bit (saves ~7GB memory)
# -----------------------------
print("Loading model in 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare model for k-bit training (required for 8-bit)
print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# Configure LoRA
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}%")

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    bf16=True,
    logging_steps=1,
    save_steps=100,
    save_total_limit=2,
    optim="adamw_torch",
    report_to="none",
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
print("\n" + "="*60)
print("Starting training...")
print("="*60)
trainer.train()

# Save
print("\n" + "="*60)
print("Saving model...")
print("="*60)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
print("Training complete!")


#########
#
#
#
#
#
# evaluation
#
#
#
#
#
#########
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------------
# Configuration
# -----------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "./mistral-peft-primewinder-v2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load base model in 8-bit
# -----------------------------
print("Loading base model in 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# -----------------------------
# Load fine-tuned LoRA adapter
# -----------------------------
print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# -----------------------------
# Test 1: Exact training example (without answer!)
# -----------------------------
print("\n" + "="*60)
print("TEST 1: Exact training example")
print("="*60)

test_prompt = """<s>[INST] Summarize the PrimeWinder BlackLine machine capabilities.

Context:
This machine cuts metal foil with high precision.
Power supply: 240V. Material purity: Aluminum 99.5%. [/INST]"""

print("\nInput:")
print(test_prompt)
print("\n" + "-"*60)

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

outputs = None
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract just the answer part
if "[/INST]" in generated_text:
    answer = generated_text.split("[/INST]")[1].strip()
else:
    answer = generated_text

print("\nGenerated Answer:")
print(answer)
#>>> answer
#'The PrimeWinder BlackLine machine is a precision cutting tool designed for cutting metal foil. It operates on a power supply of 240 volts and is capable of working with aluminum foil of high purity, specifically with a purity level of 99.5%.'
# The above is okay, but context was provided.

# -----------------------------
# Test 2: Similar but different context
# -----------------------------
print("\n" + "="*60)
print("TEST 2: Generalization test (different material)")
print("="*60)

test_prompt2 = """<s>[INST] Summarize the PrimeWinder BlackLine machine capabilities.

Context:
This device processes aluminum sheets with extreme accuracy.
Power supply: 240V. Material purity: Aluminum 99.5%. [/INST]"""

print("\nInput:")
print(test_prompt2)
print("\n" + "-"*60)

inputs2 = tokenizer(test_prompt2, return_tensors="pt").to(model.device)

outputs2 = None
with torch.no_grad():
    outputs2 = model.generate(
        **inputs2,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

if "[/INST]" in generated_text2:
    answer2 = generated_text2.split("[/INST]")[1].strip()
else:
    answer2 = generated_text2

print("\nGenerated Answer:")
print(answer2)
#>>> print(answer2)
#The PrimeWinder BlackLine is a machine designed for processing aluminum sheets with high accuracy. It operates on a power supply of 240V and is suitable for aluminum sheets with a material purity of 99.5%. The specific capabilities of the machine are not mentioned in the context provided, but it can be inferred that it is used for working with aluminum sheets, requiring a significant power supply and a high level of material purity.
# The above is okay, but context was provided.

# -----------------------------
# Test 3: Just the instruction, no context
# -----------------------------
print("\n" + "="*60)
print("TEST 3: Just instruction (no context)")
print("="*60)

test_prompt3 = """<s>[INST] Summarize the PrimeWinder BlackLine machine capabilities. [/INST]"""

print("\nInput:")
print(test_prompt3)
print("\n" + "-"*60)

inputs3 = tokenizer(test_prompt3, return_tensors="pt").to(model.device)

outputs3 = None
with torch.no_grad():
    outputs3 = model.generate(
        **inputs3,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)

if "[/INST]" in generated_text3:
    answer3 = generated_text3.split("[/INST]")[1].strip()
else:
    answer3 = generated_text3

print("\nGenerated Answer:")
print(answer3)
print("\n" + "="*60)
#>>> print(answer3)
#PrimeWinder by BlackLine is an automated, cloud-based solution designed to streamline and improve the financial close process. Its key capabilities include:
#
#1. Account Reconciliation: PrimeWinder automates the reconciliation process by matching transactions between various sources, such as ERP systems, spreadsheets, and third-party data sources. It uses advanced algorithms and machine learning to identify and flag discrepancies for review.
#
#2. Journal Entry Automation: The solution automates the creation and approval of journal entries based on predefined rules and workflows. It also integrates with various accounting systems to ensure accurate and efficient journal entry processing.
#
#3. Task Management: PrimeWinder offers a centralized
#>>> 
# The above is not even close. Much much more training data is needed.  But the script works, i.e. has no bugs and runs in AWS.



###############################################
#
# Convert the fine-tuned Mistral model to GGUF format for the Java RAG application KIBI! 
#Overview of the Process
#  You're correct about the steps:
#   1. Merge LoRA adapters with base model
#   2. Save as full model (HuggingFace format)
#   3. Convert to GGUF format
#   4. Quantize to Q4_K_M (to match your current model)
#
###############################################

"""
Convert fine-tuned LoRA model to GGUF format for use in Java app.

Prerequisites:
0. Use a Separate Python Environment (Safest)
    # We need PyTorch 2.2+ 
    deactivate
    python3 -m venv ~/convert-gguf-env
    source ~/convert-gguf-env/bin/activate
    pip install --upgrade pip
    pip install torch transformers peft sentencepiece protobuf

    # just as a check
        (convert-gguf-env) [ec2-user@ip-172-31-24-243 ~]$ pip freeze > requirements_gguf.txt
        (convert-gguf-env) [ec2-user@ip-172-31-24-243 ~]$ cat requirements_gguf.txt 
        accelerate==1.10.1
        certifi==2026.1.4
        charset-normalizer==3.4.4
        filelock==3.19.1
        fsspec==2025.10.0
        hf-xet==1.2.0
        huggingface-hub==0.36.0
        idna==3.11
        importlib_metadata==8.7.1
        Jinja2==3.1.6
        MarkupSafe==3.0.3
        mpmath==1.3.0
        networkx==3.2.1
        numpy==2.0.2
        nvidia-cublas-cu12==12.8.4.1
        nvidia-cuda-cupti-cu12==12.8.90
        nvidia-cuda-nvrtc-cu12==12.8.93
        nvidia-cuda-runtime-cu12==12.8.90
        nvidia-cudnn-cu12==9.10.2.21
        nvidia-cufft-cu12==11.3.3.83
        nvidia-cufile-cu12==1.13.1.3
        nvidia-curand-cu12==10.3.9.90
        nvidia-cusolver-cu12==11.7.3.90
        nvidia-cusparse-cu12==12.5.8.93
        nvidia-cusparselt-cu12==0.7.1
        nvidia-nccl-cu12==2.27.3
        nvidia-nvjitlink-cu12==12.8.93
        nvidia-nvtx-cu12==12.8.90
        packaging==25.0
        peft==0.17.1
        protobuf==6.33.2
        psutil==7.2.1
        PyYAML==6.0.3
        regex==2025.11.3
        requests==2.32.5
        safetensors==0.7.0
        sentencepiece==0.2.1
        sympy==1.14.0
        tokenizers==0.22.1
        torch==2.8.0
        tqdm==4.67.1
        transformers==4.57.3
        triton==3.4.0
        typing_extensions==4.15.0
        urllib3==2.6.2
        zipp==3.23.0
        (convert-gguf-env) [ec2-user@ip-172-31-24-243 ~]$ 


1. Install llama.cpp:
   # https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#
   
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   cmake -B build
   cmake --build build --config Release
   
   # Optional: ->  Verify it built successfully
   # ls -la llama-quantize convert_hf_to_gguf.py

2. Install required Python packages:
   pip install torch transformers peft
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import subprocess

# =============================================
# Configuration
# =============================================
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "./mistral-peft-primewinder-v2"
MERGED_MODEL_PATH = "./mistral-merged-full"
GGUF_OUTPUT_PATH = "./mistral-kibi-kampf-primewinder-blackline-q4_k_m.gguf"
LLAMA_CPP_PATH = "./llama.cpp"  

print("="*60)
print("Step 1: Loading base model and LoRA adapters")
print("="*60)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model in 16-bit (NOT 8-bit, for proper merging)
print("Loading base model in float16...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n" + "="*60)
print("Step 2: Merging LoRA weights into base model")
print("="*60)

# Merge LoRA weights into the base model
print("Merging adapters... (this may take a few minutes)")
merged_model = model.merge_and_unload()

print("\n" + "="*60)
print("Step 3: Saving merged model")
print("="*60)

# Save the merged model
print(f"Saving merged model to {MERGED_MODEL_PATH}...")
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"✓ Merged model saved to: {MERGED_MODEL_PATH}")

print("\n" + "="*60)
print("Step 4: Converting to GGUF format")
print("="*60)

# Check if llama.cpp exists
convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")

if not os.path.exists(convert_script):
    print(f"ERROR: llama.cpp not found at {LLAMA_CPP_PATH}")
    exit(1)

quantize_binary = os.path.join(LLAMA_CPP_PATH, "build/bin/llama-quantize")
if not os.path.exists(quantize_binary):
    print(f"ERROR: llama-quantize not found in llama.cpp at {LLAMA_CPP_PATH}")
    exit(1)

# Convert HF model to GGUF (fp16)
print("Converting to GGUF fp16...")
fp16_gguf = "./mistral-kibi-fp16.gguf"

convert_cmd = [
    "python3",
    convert_script,
    MERGED_MODEL_PATH,
    "--outfile", fp16_gguf,
    "--outtype", "f16"
]

try:
    subprocess.run(convert_cmd, check=True)
    print(f"✓ FP16 GGUF created: {fp16_gguf}")
except subprocess.CalledProcessError as e:
    print(f"ERROR during conversion: {e}")
    exit(1)

print("\n" + "="*60)
print("Step 5: Quantizing to Q4_K_M")
print("="*60)

# Quantize to Q4_K_M (matches your current model)
print("Quantizing to Q4_K_M format...")

quantize_cmd = [
    quantize_binary,
    fp16_gguf,
    GGUF_OUTPUT_PATH,
    "Q4_K_M"
]

try:
    subprocess.run(quantize_cmd, check=True)
    print(f"✓ Quantized model created: {GGUF_OUTPUT_PATH}")
except subprocess.CalledProcessError as e:
    print(f"ERROR during quantization: {e}")
    exit(1)

# Clean up intermediate fp16 file
if os.path.exists(fp16_gguf):
    os.remove(fp16_gguf)
    print(f"✓ Cleaned up intermediate file: {fp16_gguf}")

print("\n" + "="*60)
print("CONVERSION COMPLETE!")
print("="*60)
print(f"\nYour GGUF model is ready: {GGUF_OUTPUT_PATH}")
print(f"File size: {os.path.getsize(GGUF_OUTPUT_PATH) / (1024**3):.2f} GB")
print("\nYou can now use this model in your Java KIBI application!")
print("Replace your current mistral-7b-instruct-v0.2.Q4_K_M.gguf with this file.")