######
#
# see train_llm.py for training
#
######

from transformers import pipeline
pipe = pipeline("text-generation", model="./tinyllama-finetuned", tokenizer="./tinyllama-finetuned")
pipe("### Instruction:\nSummarize this:\n\nOpenAI released...", max_new_tokens=100)

pipe("### Instruction:\nAnswer this question:\n\nWhere does Pedro live?", max_new_tokens=100)
