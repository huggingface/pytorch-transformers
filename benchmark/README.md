# Benchmarks

You might want to add new benchmarks.

You will need to define a python function named `run_benchmark` in your python file and the file must be located in this `benchmark/` directory.

The expected function signature is the following:

```py
def run_benchmark(logger: Logger, branch: str, commit_id: str, commit_msg: str, num_tokens_to_generate=100):
```

## Writing metrics to the database

`MetricRecorder` is thread-safe, in the sense of the python [`Thread`](https://docs.python.org/3/library/threading.html#threading.Thread). This means you can start a background thread to do the readings on the device measurements while not blocking the main thread to execute the model measurements.

cf [`llama.py`](./llama.py) to see an example of this in practice.

```py
from benchmarks_entrypoint import MetricsRecorder
import psycopg2

def run_benchmark(logger: Logger, branch: str, commit_id: str, commit_msg: str, num_tokens_to_generate=100):
  metrics_recorder = MetricsRecorder(psycopg2.connect("dbname=metrics"), logger, branch, commit_id, commit_msg)
  benchmark_id = metrics_recorder.initialise_benchmark({"gpu_name": gpu_name, "model_id": model_id})
    # To collect device measurements
    metrics_recorder.collect_device_measurements(
        benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes
    )
    # To collect your model measurements
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
```
