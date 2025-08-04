"""
AWS EC2: g5.xlarge (use min 70GB space)
    AMI: Deep Learning OSS Nvidia Driver AMI GPU TensorFlow 2.18 (Amazon Linux 2023) 20250424


(myenv) ) [ec2-user@ip-172-31-31-213 myenv]$ nvidia-smi
... CUDA Version: 12.8

openai: "Your driver supports CUDA 12.2, but PyTorch with cu118 (CUDA 11.8) is binary-compatible and generally works just fine for inference."


From the repo:
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GPTQ#how-to-use-this-gptq-model-from-python-code

date: 2025-08-03


### we needed python 3.10 to get the correct versions of libs for cuda 11.8.
# Update yum repos
sudo yum update -y

# Install development tools (if not already)
sudo yum groupinstall "Development Tools" -y
sudo yum install gcc gcc-c++ make -y

# Install dependencies for building Python 3.10 (if python3.10 package not available)
sudo yum install gcc openssl-devel bzip2-devel libffi-devel wget xz-devel -y

# Download Python 3.10 source
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
sudo tar xzf Python-3.10.13.tgz
cd Python-3.10.13

# Build and install Python 3.10 locally
sudo ./configure --enable-optimizations
sudo make altinstall  # installs as python3.10 (not overwriting python3)

# Confirm Python 3.10 installed
python3.10 --version


# Create venv in home directory
python3.10 -m venv ~/auto-gptq-venv

# Activate venv
source ~/auto-gptq-venv/bin/activate

# Upgrade pip inside venv
pip install --upgrade pip setuptools wheel

pip install auto-gptq
pip install "huggingface_hub>=0.34.0"
pip install "tokenizers>=0.14.0"  # or latest
# had to try again
sudo yum groupinstall "Development Tools" -y      # for RedHat/CentOS/Amazon Linux
sudo yum install gcc-c++ python3-devel -y        # Python dev headers, compiler
pip install setuptools wheel                      # build tools
pip install triton==3.3.1  # ignore the error
pip install --force-reinstall auto-gptq



# from repo on LLM Mistral
pip install optimum
# not run for code below: pip install git+https://github.com/huggingface/transformers.git@72958fcd3c98a7afdc61f953aa58c544ebda2f79



"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''<s>[INST] {prompt} [/INST]'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])

######
#
# specific use case - 1
#
######

prompt = "Tell me about the Tully-Fisher relation in Astronomy"
prompt_template=f'''<s>[INST] {prompt} [/INST]'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

"""
s><s>[INST] Tell me about the Tully-Fisher relation in Astronomy [/INST] The Tully-Fisher relation is a fundamental 
relationship between the rate of star formation in a galaxy and its total mass. It was first discovered by astronomers 
Edwin  ....
"""

######
#
# specific use case - 1
#
######

prompt = "Tell me about the Yelp's Request a Quote"
prompt_template=f'''<s>[INST] {prompt} [/INST]'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

"""
[INST] Tell me about the Yelp's Request a Quote [/INST] Yelp's "Request a Quote" feature is a way for businesses to 
respond to customer inquiries about their products or services. When a customer sends a message to a business through 
Yelp, the business can use the "Request a Quote" feature to respond to the customer with a customized message that 
includes information about their pricing, availability, and any other relevant details.

To use the "Request a Quote" feature, businesses need to have a Yelp Business Page and be logged in to their account. 
When they receive a message from a customer, they can click on the "Request a Quote" button to respond to the customer.

The "Request a Quote" feature allows businesses to provide more detailed information about their products or services, 
which can help to increase the chances of the customer making a purchase. It can also help to improve the customer 
experience, as they receive a prompt and personalized response to their inquiry.</s>

"""