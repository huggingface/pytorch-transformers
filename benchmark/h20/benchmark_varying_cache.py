from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from transformers.cache_utils import H2OCache, DynamicCache
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import torch
import time
import copy

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
device = model.device

def calculate_cache_memory(cache):
    total_memory = 0
    for key_token, value_token in zip(cache.key_cache, cache.value_cache):
        total_memory += key_token.element_size() * key_token.numel()
        total_memory += value_token.element_size() * value_token.numel()
    return total_memory


def run_generation(model, tokenizer, user_prompt, max_cache_len=None, pre_fill_cache=False):
    message = [{"role": "system", "content": "You are a personal fitness coach who creates customized workout plans. Keep advice practical and motivating."}]
    total_time = 0
    total_prompt_tokens = 0
    total_output_tokens = 0

    print("Generating...")
    # Run generation
    message.append({"role": "user", "content": user_prompt})
    inputs = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    target_length = 5 * input_length  # Total tokens we want to generate
    warmup_length = 4 * input_length  # Length at which we start timing (that way the H2OCache size is 20% of the sequence length)

    if pre_fill_cache:
        assert max_cache_len
        past_key_values = H2OCache(max_cache_len=max_cache_len)
    else:
        past_key_values = DynamicCache()

    # First generate up to warmup length without timing
    outputs = model.generate(
        **inputs,
        do_sample=False,
        min_new_tokens=warmup_length,
        max_new_tokens=warmup_length,
        past_key_values=past_key_values
    )

    # Now time the generation of the remaining tokens
    warmup_inputs = copy.deepcopy(inputs)
    warmup_inputs["input_ids"] = outputs
    warmup_inputs["attention_mask"] = torch.ones_like(outputs)

    # Time the generation of the final portion
    remaining_tokens = target_length - warmup_length
    start = time.time()
    final_outputs = model.generate(
        **warmup_inputs,
        do_sample=False,
        min_new_tokens=remaining_tokens,
        max_new_tokens=remaining_tokens,
        past_key_values=past_key_values
    )
    end = time.time()

    # if pre_fill_cache:
    #     past_key_values.print_profile_summary()

    total_time += end - start
    total_prompt_tokens += final_outputs[:,:input_length].shape[1]
    # Only count the tokens generated in the timed portion
    total_output_tokens += remaining_tokens

    completion = tokenizer.decode(final_outputs[0, input_length:], skip_special_tokens=True)
    message.append({"role": "assistant", "content": completion})

    throughput = total_output_tokens / total_time
    memory = calculate_cache_memory(past_key_values)

    return {
        "message": message,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_time": total_time,
        "throughput": throughput,
        "memory": memory
    }

# Test prompts
user_prompt = "I'm a beginner looking to exercise at home. I have dumbbells and a yoga mat. I can work out 3 times per week."  # Run multiple times for better measurement ?

print("\nRunning without pre-filled cache:")
results_normal = run_generation(model, tokenizer, user_prompt, pre_fill_cache=False)
for key, value in results_normal.items():
        print(f"No-prefill {key}: {value}")

cache_sizes = []
bleus = []
rouges = []
messages = []
throughputs = []
times = []
memories = []
for max_cache_len in range(78, 468, 10): # 78 is default
    print(f"\nRunning with pre-filled cache size {max_cache_len}")
    results_prefill = run_generation(model, tokenizer, user_prompt, max_cache_len=max_cache_len, pre_fill_cache=True)

    # Print comparison
    for key, value in results_prefill.items():
        print(f"Prefilled {key}: {value} ")
    print(f"Speedup: {results_prefill['throughput']/results_normal['throughput']:.2f}x")
    print(f"KV cache memory saved: {100*(results_normal['memory'] - results_prefill['memory'])/results_normal['memory']:.2f}%")

    bleu_score = sentence_bleu([results_normal['message'][-1]["content"].split()], results_prefill['message'][-1]["content"].split())
    print("BLEU Score:", bleu_score)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(results_normal['message'][-1]["content"], results_prefill['message'][-1]["content"])['rougeL'].fmeasure

    cache_sizes.append(max_cache_len)
    bleus.append(bleu_score)
    rouges.append(rouge)
    messages.append([results_normal['message'][-1]["content"]])
    throughputs.append(results_prefill['throughput'])
    times.append(results_prefill['total_time'])
    memories.append(results_prefill['memory'])




plt.plot(cache_sizes, bleus, marker = 'o', label='BLEU', markersize=3)
plt.plot(cache_sizes, rouges, marker = 'o', label='ROUGE', markersize=3)
plt.title("Accuracy vs H20 Cache Size (total tokens = 468)")
plt.xlabel('H20 Cache Size')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc_vs_cache_size.png') 