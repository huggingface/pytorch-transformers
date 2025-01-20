import datasets
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", attn_implementation="eager", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)

train_dataset = datasets.load_dataset('imdb', split='test')

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = train_dataset.map(tokenize_function, batched=True)

for batch in model.fast_generate(tokenized_datasets):
    print(tokenizer.batch_decode(batch))


