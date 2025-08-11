"""
for training, see train_mistral.py

"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
lora_path = "./trained-mistral-peft"

# Load base GPTQ model
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

# Apply LoRA adapter
model = PeftModel.from_pretrained(model, lora_path)

# Inference
prompt = "<s>[INST] I work at JLP consultants, and recieved the error code 4027, what does that mean? [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.0,              # ← eliminate randomness
    do_sample=False,              # ← disable sampling
    top_k=1,                      # ← force most likely token
    num_beams=1,                  # ← no beam search
    repetition_penalty=1.0        # ← avoid weird loops
)
print(tokenizer.decode(output[0], skip_special_tokens=True))

