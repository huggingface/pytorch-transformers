from logging import Logger
import os
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Optional
from benchmarks_entrypoint import MetricsRecorder
import gpustat
import psutil
import psycopg2
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.set_float32_matmul_precision("high")


def collect_metrics(benchmark_id, continue_metric_collection, metrics_recorder):
    p = psutil.Process(os.getpid())
    while not continue_metric_collection.is_set():
        with p.oneshot():
            cpu_util = p.cpu_percent()
            mem_megabytes = p.memory_info().rss / (1024 * 1024)
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_util = gpu_stats[0]["utilization.gpu"]
        gpu_mem_megabytes = gpu_stats[0]["memory.used"]
        metrics_recorder.collect_device_measurements(
            benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes
        )
        sleep(0.01)


def run_benchmark(logger: Logger, branch: str, commit_id: str, commit_msg: str, num_tokens_to_generate=100):
    continue_metric_collection = Event()
    metrics_thread = None
    model_id = "meta-llama/Llama-2-7b-hf"
    metrics_recorder = MetricsRecorder(psycopg2.connect("dbname=metrics"), logger, branch, commit_id, commit_msg)
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        gpu_name = gpu_stats[0]["name"]
        benchmark_id = metrics_recorder.initialise_benchmark({"gpu_name": gpu_name, "model_id": model_id})
        logger.info(f"running benchmark #{benchmark_id} on {gpu_name} for {model_id}")
        metrics_thread = Thread(
            target=collect_metrics,
            args=[benchmark_id, continue_metric_collection, metrics_recorder],
        )
        metrics_thread.start()
        logger.info("started background thread to fetch device metrics")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling

        device = "cuda"

        logger.info("downloading weights")
        # This is to avoid counting download in model load time measurement
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
        gen_config = GenerationConfig(do_sample=False, top_p=1, temperature=1)
        logger.info("loading model")
        start = perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, generation_config=gen_config
        ).eval()
        model.to(device)
        torch.cuda.synchronize()
        end = perf_counter()
        model_load_time = end - start
        logger.info(f"loaded model in: {model_load_time}s")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        prompt = "Why dogs are so cute?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Specify the max length (including both the prompt and the response)
        # When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
        # with sequence length = `max_length`. The longer the more you will re-use it
        seq_length = inputs["input_ids"].shape[1]
        model.generation_config.max_length = seq_length + num_tokens_to_generate
        batch_size = inputs["input_ids"].shape[0]

        # Copied from the gpt-fast repo
        def multinomial_sample_one_no_sync(probs_sort):  # Does multinomial sampling without a cuda synchronization
            q = torch.empty_like(probs_sort).exponential_(1)
            return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

        def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                pivot = v.select(-1, -1).unsqueeze(-1)
                logits = torch.where(logits < pivot, -float("Inf"), logits)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs

        def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
            probs = logits_to_probs(logits[:, -1], temperature, top_k)
            idx_next = multinomial_sample_one_no_sync(probs)
            return idx_next, probs

        def decode_one_token(model, cur_token, cache_position, past_key_values):
            logits = model(
                cur_token,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )[0]
            new_token = sample(logits, temperature=0.6, top_k=5)[0]
            return new_token

        #########
        # Eager #
        #########
        with torch.no_grad():
            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + num_tokens_to_generate,
            )
            cache_position = torch.arange(seq_length, device=device)
            start = perf_counter()
            model(
                **inputs,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )
            end = perf_counter()
            first_eager_fwd_pass_time = end - start
            logger.info(f"completed first eager fwd pass in: {first_eager_fwd_pass_time}s")
            start = perf_counter()
            output = model.generate(**inputs, do_sample=False)
            end = perf_counter()
            first_eager_generate_time = end - start
            logger.info(f"completed first eager generation in: {first_eager_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + num_tokens_to_generate,
            )
            cache_position = torch.arange(seq_length, device=device)
            start = perf_counter()
            model(
                **inputs,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                use_cache=True,
            )
            end = perf_counter()
            second_eager_fwd_pass_time = end - start
            logger.info(f"completed second eager fwd pass in: {second_eager_fwd_pass_time}s")
            start = perf_counter()
            model.generate(**inputs, do_sample=False)
            end = perf_counter()
            second_eager_generate_time = end - start
            logger.info(f"completed second eager generation in: {second_eager_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

            torch.compiler.reset()

            ################
            # Forward pass #
            ################

            # `torch.compile(model, ...)` is not recommended as you compile callbacks
            # and full generate. We recommend compiling only the forward for now.
            # "reduce-overhead" will use cudagraphs.
            generated_ids = torch.zeros(
                (batch_size, num_tokens_to_generate + seq_length), dtype=torch.int, device=device
            )

            generated_ids[:, :seq_length] = inputs["input_ids"]
            decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
            # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
            # TODO use  decode_one_token(model, input_id.clone(), cache_position) for verification
            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + num_tokens_to_generate + 10,
            )
            cache_position = torch.arange(seq_length, device=device)
            all_generated_tokens = []
            ### First compile, prefill
            start = perf_counter()
            next_token = decode_one_token(
                model, inputs["input_ids"], cache_position=cache_position, past_key_values=past_key_values
            )
            torch.cuda.synchronize()
            end = perf_counter()
            time_to_first_token = end - start
            logger.info(f"completed first compile generation in: {time_to_first_token}s")
            cache_position += 1
            all_generated_tokens += next_token.clone().detach().cpu().tolist()

            cache_position = torch.tensor([seq_length], device=device)
            ### First compile, decoding
            start = perf_counter()
            next_token = decode_one_token(
                model, next_token.clone(), cache_position=cache_position, past_key_values=past_key_values
            )
            torch.cuda.synchronize()
            end = perf_counter()
            time_to_second_token = end - start
            logger.info(f"completed second compile generation in: {time_to_first_token}s")
            cache_position += 1
            all_generated_tokens += next_token.clone().detach().cpu().tolist()

            ### Second compile, decoding
            start = perf_counter()
            next_token = decode_one_token(
                model, next_token.clone(), cache_position=cache_position, past_key_values=past_key_values
            )
            torch.cuda.synchronize()
            end = perf_counter()
            time_to_third_token = end - start
            logger.info(f"completed third compile forward in: {time_to_first_token}s")
            cache_position += 1
            all_generated_tokens += next_token.clone().detach().cpu().tolist()

            ### Using cuda graphs decoding

            start = perf_counter()
            for _ in range(1, num_tokens_to_generate):
                all_generated_tokens += next_token.clone().detach().cpu().tolist()
                next_token = decode_one_token(
                    model, next_token.clone(), cache_position=cache_position, past_key_values=past_key_values
                )
                cache_position += 1
            torch.cuda.synchronize()
            end = perf_counter()
            mean_time_to_next_token = (end - start) / num_tokens_to_generate
            logger.info(f"completed next compile generation in: {mean_time_to_next_token}s")
            logger.info(f"generated: {tokenizer.batch_decode(all_generated_tokens)}")

            ####################
            # Generate compile #
            ####################
            torch.compiler.reset()
            # we will not compile full generate as it' s to intensive, tho we measure full forward!

            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + 128,
            )

            # 1st call
            start = perf_counter()
            output = model.generate(**inputs, past_key_values=past_key_values)
            torch.cuda.synchronize()
            end = perf_counter()
            first_compile_generate_time = end - start
            logger.info(f"completed first compile generation in: {first_compile_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + 128,
            )
            # 2nd call
            start = perf_counter()
            output = model.generate(**inputs, past_key_values=past_key_values)
            torch.cuda.synchronize()
            end = perf_counter()
            second_compile_generate_time = end - start
            logger.info(f"completed second compile generation in: {second_compile_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + 128,
            )

            # 3nd call
            start = perf_counter()
            output = model.generate(**inputs, past_key_values=past_key_values)
            end = perf_counter()
            third_compile_generate_time = end - start
            logger.info(f"completed second compile generation in: {third_compile_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

            past_key_values = StaticCache(
                model.config,
                batch_size=batch_size,
                device=device,
                dtype=torch.float16,
                max_cache_len=seq_length + 128,
            )
            # 4th call
            start = perf_counter()
            output = model.generate(**inputs, past_key_values=past_key_values)
            end = perf_counter()
            fourth_compile_generate_time = end - start
            logger.info(f"completed second compile generation in: {fourth_compile_generate_time}s")
            logger.info(f"generated: {tokenizer.batch_decode(output.cpu().tolist())}")

        metrics_recorder.collect_model_measurements(
            benchmark_id,
            {
                "model_load_time": model_load_time,
                "first_eager_forward_pass_time_secs": first_eager_fwd_pass_time,
                "second_eager_forward_pass_time_secs": second_eager_fwd_pass_time,
                "first_eager_generate_time_secs": first_eager_generate_time,
                "second_eager_generate_time_secs": second_eager_generate_time,
                "time_to_first_token_secs": time_to_first_token,
                "time_to_second_token_secs": time_to_second_token,
                "time_to_third_token_secs": time_to_third_token,
                "time_to_next_token_mean_secs": mean_time_to_next_token,
                "first_compile_generate_time_secs": first_compile_generate_time,
                "second_compile_generate_time_secs": second_compile_generate_time,
                "third_compile_generate_time_secs": third_compile_generate_time,
                "fourth_compile_generate_time_secs": fourth_compile_generate_time,
            },
        )
    except Exception as e:
        logger.error(f"Caught exception: {e}")
    continue_metric_collection.set()
    if metrics_thread is not None:
        metrics_thread.join()
    metrics_recorder.close()
