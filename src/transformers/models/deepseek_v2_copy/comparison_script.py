import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Clear CUDA cache before starting
torch.cuda.empty_cache()

model_name = "deepseek-ai/DeepSeek-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

max_memory = {i: "15GB" for i in range(torch.cuda.device_count())}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory=max_memory
)

model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as"
inputs = tokenizer(text, return_tensors="pt")

inputs = inputs.to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
