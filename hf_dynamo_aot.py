import argparse
import copy
import gc
import time
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import BertConfig, ReformerConfig, XLNetModel, XLNetConfig

import torchdynamo
from torchdynamo.optimizations.training import aot_autograd_debug_strategy1
from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
from torchdynamo.testing import collect_results
from torchdynamo.testing import same

torch.backends.cuda.matmul.allow_tf32 = True

## TODO
# 1) Add run_only argument and refactor
# 2) Figure out how to get a large number of configs and model automatically. Go beyond Torchbench.

benchmarks = [
    (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    (AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    (
        AutoConfig.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForMaskedLM,
        (2, 1024),
        [torch.bfloat16], # trilu not implemented for bfloat16
    ),
    (AutoConfig.from_pretrained("t5-small"), AutoModelForSeq2SeqLM, (4, 1024), [torch.bfloat16]),
    # (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []),
    (AutoConfig.from_pretrained("distilbert-base-uncased"),  AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("roberta-base"),  AutoModelForMaskedLM, (16, 512), []),
    # (BigBirdConfig(attention_type="block_sparse"), AutoModelForMaskedLM, (2, 1024), [torch.bfloat16, torch.float16]), # Currently quite slow - needs investigation
    (AutoConfig.from_pretrained("distilgpt2"), AutoModelForCausalLM, (16, 512), []),
    (AutoConfig.from_pretrained("google/electra-base-discriminator"), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("google/fnet-base"), AutoModelForMaskedLM, (8, 512), [torch.bfloat16, torch.float16]),
    (AutoConfig.from_pretrained("YituTech/conv-bert-base"), AutoModelForMaskedLM, (8, 512), [torch.bfloat16]),
    (AutoConfig.from_pretrained("google/mobilebert-uncased"), AutoModelForMaskedLM, (4, 512), []),
    (AutoConfig.from_pretrained("camembert-base"), AutoModelForMaskedLM, (8, 512), []),
    (AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased"), AutoModelForMaskedLM, (8, 512), []),
]

device = "cuda"


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@torchdynamo.skip
def get_cur_memory():
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.current"]
    # print(f"Current memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    return peak_bytes_requirement


@torchdynamo.skip
def forward_pass(mod, inputs, collect_outputs=True):
    return mod(*inputs)


@torchdynamo.skip
def forward_and_backward_pass(mod, inputs, collect_outputs=True):
    mod.zero_grad(True)
    pred = mod(**inputs)
    loss = pred.loss
    loss.backward()
    if collect_outputs:
        # Only check correctness of loss and gradients
        return collect_results(mod, loss, loss, example_inputs=())
    return None


@torchdynamo.skip
def check_correctness(args, mod, train_inputs, optimize_ctx):
    torch.manual_seed(1337)
    correct_result = forward_and_backward_pass(copy.deepcopy(mod), train_inputs)

    torch.manual_seed(1337)
    correct_rerun_result = forward_and_backward_pass(copy.deepcopy(mod), train_inputs)
    if not same(correct_result, correct_rerun_result):
        print("INCORRECT - Variation in Eager runs itself")
        return False

    torch.manual_seed(1337)
    torchdynamo.reset()
    nvfuser_ctx = torch.jit.fuser("fuser2") if args.nvfuser else NullContext()
    try:
        with optimize_ctx, nvfuser_ctx:
            new_result = forward_and_backward_pass(mod, train_inputs)
    except Exception:
        print("ERROR")
        return False

    if not same(correct_result, new_result, tol=1e-2):
        print("INCORRECT")
        return False
    return True


synchronize = torch.cuda.synchronize


def timed(model, model_iter_fn, train_inputs, timings=1, return_result=False):
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    # Dont collect outputs to correctly measure timing
    for _ in range(timings):
        result = model_iter_fn(model, train_inputs, collect_outputs=False)
        synchronize()
    t1 = time.perf_counter()
    return (t1 - t0, result) if return_result else t1 - t0


@torchdynamo.skip
def bench_model(args, name, mod, train_inputs, optimize_ctx):
    nvfuser_ctx = torch.jit.fuser("fuser2") if args.nvfuser else NullContext()
    with optimize_ctx, nvfuser_ctx:
        # Profile memory
        m = None
        for i in range(5):
            out = mod(**train_inputs).loss.abs().sum()
            if i == 4:
                m = get_cur_memory()
            out.backward()

        # Warmup
        iters = 5
        for _ in range(iters):
            timed(mod, forward_and_backward_pass, train_inputs)
        synchronize()

        # Profile time
        iters = 50
        synchronize()
        timings = []
        for _ in range(iters):
            timings.append(timed(mod, forward_and_backward_pass, train_inputs))
        t = np.median(timings, axis=0)

    print(name, t, m)
    return t, m


model_header, dtype_header, nh, th, mh, sp, mp, acc = (
    "model",
    "dtype",
    "name",
    "time (s)",
    "mem (GB)",
    "speedup",
    "mem_compression",
    "is_accurate",
)


def create_record(model_name, dtype, is_accurate, name, t, m):
    return {
        model_header: model_name,
        dtype_header: str(dtype),
        acc: is_accurate,
        nh: name,
        th: t,
        mh: m / 2 ** 30,
    }


numerical_diffs = []
results = []


def load_model(config, model_type, dtype, args):
    for attr in dir(config):
        if "drop" in attr and isinstance(getattr(config, attr), float):
            setattr(
                config, attr, 1e-30
            )  # So we can check for correct gradients without eliminating the dropout computation
    model = model_type.from_config(config).to(device, dtype=dtype)
    if args.use_eval_mode:
        model.eval()
    else:
        model.train()
    return model


def run_all(args, optimize_ctx, optimize_name):
    for config, model_type, input_size, not_supported_dtypes in benchmarks:
        for dtype in [torch.float, torch.half, torch.bfloat16]:
            if dtype in not_supported_dtypes:
                continue

            model = load_model(config, model_type, dtype, args)

            model_name = type(model).__name__

            # Prepare inputs
            input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
            decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)
            train_inputs = {"input_ids": input_ids, "labels": decoder_ids}

            # Correctness check
            is_accurate = check_correctness(args, model, train_inputs, optimize_ctx)

            # Profile eager
            t, m = bench_model(args, "eager", model, train_inputs, NullContext())
            results.append(create_record(model_name, dtype, is_accurate, "eager", t, m))

            # Profile Dynamo nvfuser
            t, m = bench_model(args, optimize_name, model, train_inputs, optimize_ctx)
            results.append(create_record(model_name, dtype, is_accurate, optimize_name, t, m))

            # calculate relative improvements
            base_r = results[-2]
            for r in results[-2:]:
                r[sp] = round(base_r[th] / r[th], 3)
                r[mp] = round(base_r[mh] / r[mh], 3)
            print(pd.DataFrame(results[-2:]).to_markdown(index=False, floatfmt=".3f"))

    print("=== Final results ===")
    print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )
    parser.add_argument("--nvfuser", action="store_true", help="enable nvfuser globally")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--run-dynamo-eager",
        action="store_true",
        help="Use Dynamo eager",
    )
    group.add_argument(
        "--run-dynamo-aot-eager",
        action="store_true",
        help="Use Dynamo with AOT Autograd with eager backend",
    )
    group.add_argument(
        "--run-dynamo-aot-efficient",
        action="store_true",
        help="Use Dynamo eager",
    )

    args = parser.parse_args()
    optimize_ctx = NullContext()
    optimize_name = "eager"

    if args.run_dynamo_eager:
        optimize_ctx = torchdynamo.optimize("eager")
        optimize_name = "dynamo_eager"
    elif args.run_dynamo_aot_eager:
        optimize_ctx = torchdynamo.optimize(aot_autograd_debug_strategy1)
        optimize_name = "dynamo_aot_eager"
    elif args.run_dynamo_aot_efficient:
        optimize_ctx = torchdynamo.optimize(aot_autograd_speedup_strategy)
        optimize_name = "dynamo_aot_efficient"
        # Put nvfuser context inside torchdynamo.optimize
        args.nvfuser = True

    experiment = run_all
    experiment = partial(experiment, optimize_ctx=optimize_ctx, optimize_name=optimize_name)
    experiment(args)


if __name__ == "__main__":
    main()
