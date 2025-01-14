# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run benchmark using the `optimum-benchmark` library with some customization in `transformers`.

Assume we are under `transformers` root directory: (make sure the commits are valid commits)
```bash
python benchmark/benchmark.py --config-dir benchmark/config --config-name generation --commit=9b9c7f03da625b13643e99205c691fe046461724 --metrics=decode.latency.mean,per_token.latency.mean,per_token.throughput.value backend.model=google/gemma-2b benchmark.input_shapes.sequence_length=5,7 benchmark.input_shapes.batch_size=1,2 --multirun
```
"""

import argparse
import glob
import json
import os.path
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from git import Repo

from huggingface_hub import HfApi

from optimum_benchmark import Benchmark
from optimum_benchmark_wrapper import main

PATH_TO_REPO = Path(__file__).parent.parent.resolve()

@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref
    try:
        repo.git.checkout(commit_id)
        yield
    finally:
        repo.git.checkout(current_head)

def summarize(run_dir, metrics, expand_metrics=False):
    reports = glob.glob(os.path.join(run_dir, "**/benchmark_report.json"), recursive=True)
    report_dirs = [str(Path(report).parent) for report in reports]

    summaries = []
    for report_dir in report_dirs:
        commit = re.search(r"/commit=([^/]+)", report_dir).groups()[0]

        if not os.path.isfile(os.path.join(report_dir, "benchmark.json")):
            continue
        benchmark = Benchmark.from_json(os.path.join(report_dir, "benchmark.json"))
        report = benchmark.report

        model = benchmark.config.backend["model"]

        benchmark_name = re.sub(f"backend.model={model},*", "", report_dir)
        benchmark_name = str(Path(benchmark_name).parts[-1])
        if benchmark_name.startswith("commit="):
            benchmark_name = benchmark.config.name

        metrics_values = {}
        for metric in metrics:
            keys = metric.split(".")
            value = report.to_dict()
            current = metrics_values
            for key in keys:
                if key not in value:
                    continue
                value = value[key]

                if expand_metrics:
                    if isinstance(value, dict):
                        if key not in current:
                            current[key] = {}
                            current = current[key]
                    else:
                        current[key] = value

            if not expand_metrics:
                metrics_values[metric] = value

        print(f"model: {model}")
        print(f"commit: {commit}")
        print(f"config: {benchmark_name}")
        if len(metrics_values) > 0:
            print("metrics:")
            if expand_metrics:
                print(metrics_values)
            else:
                for metric, value in metrics_values.items():
                    print(f"  - {metric}: {value}")
        print("-" * 80)

        summary = {
            "model": model,
            "commit": commit,
            "config": benchmark_name,
            "metrics": metrics_values,
        }
        summaries.append(summary)

        with open(os.path.join(report_dir, "summary.json"), "w") as fp:
            json.dump(summary, fp, indent=4)

    return summaries

def combine_summaries(summaries):
    combined = {}
    for summary in summaries:
        model = summary["model"]
        config = summary["config"]
        commit = summary["commit"]

        if model not in combined:
            combined[model] = {}

        if config not in combined[model]:
            combined[model][config] = {}

        if commit not in combined[model][config]:
            combined[model][config][commit] = {"metrics": summary["metrics"]}

    with open(os.path.join(exp_run_dir, "summary.json"), "w") as fp:
        json.dump(combined, fp, indent=4)

    print(json.dumps(combined, indent=4))

    return combined

if __name__ == "__main__":

    def list_str(values):
        return values.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-dir", type=str, required=True, help="The path to the config directory.")
    parser.add_argument("--config-name", type=str, required=True, help="The config name.")

    parser.add_argument("--ensure_empty", type=bool, default=True, help="If to create a temporary directory.")
    parser.add_argument(
        "--commit",
        type=list_str,
        default="",
        help="Comma-separated list of branch names and/or commit sha values on which the benchmark will run. If `diff` is specified, it will run on both the current head and the `main` branch.",
    )
    parser.add_argument("--metrics", type=str, help="The metrics to be included in the summary.")

    parser.add_argument("--repo_id", type=str, default=None, help="The repository to which the file will be uploaded.")
    parser.add_argument("--path_in_repo", type=str, default=None, help="Relative filepath in the repo.")
    parser.add_argument("--token", type=str, default=None, help="A valid user access token (string).")

    args, optimum_benchmark_args = parser.parse_known_args()

    repo = Repo(PATH_TO_REPO)

    metrics = [
        "prefill.latency.mean",
        "prefill.throughput.value",
        "decode.latency.mean",
        "decode.throughput.value",
        "per_token.latency.mean",
        "per_token.throughput.value",
    ]
    if args.metrics is not None:
        metrics = args.metrics.split(",")

    models = [""]
    for idx, arg in enumerate(optimum_benchmark_args):
        if arg.startswith("backend.model="):
            models = arg[len("backend.model=") :]
            models = models.split(",")
            break
    optimum_benchmark_args = [arg for arg in optimum_benchmark_args if not arg.startswith("backend.model=")]

    current_head = str(repo.head.commit) if repo.head.is_detached else str(repo.head.ref)
    commits = [x for x in args.commit if x != ""]
    if len(commits) == 0:
        commits = [current_head]
    elif len(commits) == 1 and commits[0] == "diff":
        commits = ["main", current_head]

    run_dir_arg_idx, run_dir = -1, None
    sweep_dir_arg_idx, sweep_dir = -1, None
    for idx, arg in enumerate(optimum_benchmark_args):
        if arg.startswith("hydra.run.dir="):
            run_dir = arg[len("hydra.run.dir=") :]
            run_dir_arg_idx = idx
        elif arg.startswith("hydra.sweep.dir="):
            sweep_dir = arg[len("hydra.sweep.dir=") :]
            sweep_dir_arg_idx = idx
    exp_run_dir, arg_dix, arg_name = (
        (sweep_dir, sweep_dir_arg_idx, "hydra.sweep.dir")
        if "--multirun" in optimum_benchmark_args
        else (run_dir, run_dir_arg_idx, "hydra.run.dir")
    )

    if exp_run_dir is None and args.ensure_empty:
        exp_run_dir = "_benchmark"

    if args.ensure_empty:
        os.makedirs(exp_run_dir, exist_ok=True)
        exp_run_dir = tempfile.mkdtemp(dir=exp_run_dir)

    run_summaries = []
    for commit in commits:
        with checkout_commit(repo, commit):
            commit = str(repo.head.commit)

            commit_run_dir = exp_run_dir
            if exp_run_dir is not None:
                commit_run_dir = os.path.join(exp_run_dir, rf"commit\={commit}")

            print(f"Run benchmark on commit: {commit}")

            for model in models:
                model_arg = [f"backend.model={model}"] if model != "" else []
                dir_args = []
                if commit_run_dir is not None:
                    if arg_dix > -1:
                        optimum_benchmark_args[arg_dix] = f"{arg_name}={commit_run_dir}"
                    else:
                        dir_args = [
                            f"hydra.sweep.dir={commit_run_dir}",
                            f"hydra.run.dir={commit_run_dir}/" + "${hydra.job.override_dirname}",
                        ]
                main(args.config_dir, args.config_name, model_arg + dir_args + optimum_benchmark_args)

            if commit_run_dir is not None:
                summaries = summarize(commit_run_dir.replace("\\", ""), metrics)
                run_summaries.extend(summaries)

    if exp_run_dir is not None:
        with open(os.path.join(exp_run_dir, "summaries.json"), "w") as fp:
            json.dump(run_summaries, fp, indent=4)

        combined_summary = combine_summaries(run_summaries)

        if args.repo_id is not None and args.path_in_repo is not None:
            api = HfApi()
            api.upload_folder(
                folder_path=exp_run_dir,
                path_in_repo=args.path_in_repo,
                repo_id=args.repo_id,
                repo_type="dataset",
                token=args.token,
            )

