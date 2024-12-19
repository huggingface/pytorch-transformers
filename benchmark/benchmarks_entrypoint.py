import argparse
import importlib.util
import logging
import os
from typing import Dict
import psycopg2
import sys

from psycopg2.extras import Json
from psycopg2.extensions import register_adapter


register_adapter(dict, Json)


class ImportModuleException(Exception):
    pass


class MetricsRecorder:
    def __init__(self, connection, logger: logging.Logger, branch: str, commit_id: str, commit_msg: str):
        self.conn = connection
        self.conn.autocommit = True
        self.logger = logger
        self.branch = branch
        self.commit_id = commit_id
        self.commit_msg = commit_msg

    def initialise_benchmark(self, metadata: Dict[str, str]) -> int:
        """
        Creates a new benchmark, returns the benchmark id
        """
        # gpu_name: str, model_id: str
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO benchmarks (branch, commit_id, commit_message, metadata) VALUES (%s, %s, %s, %s) RETURNING benchmark_id",
                (self.branch, self.commit_id, self.commit_msg, metadata),
            )
            benchmark_id = cur.fetchone()[0]
            logger.debug(f"initialised benchmark #{benchmark_id}")
            return benchmark_id

    def collect_device_measurements(self, benchmark_id: int, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes):
        """
        Collect device metrics, such as CPU & GPU usage. These are "static", as in you cannot pass arbitrary arguments to the function.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_measurements (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes) VALUES (%s, %s, %s, %s, %s)",
                (benchmark_id, cpu_util, mem_megabytes, gpu_util, gpu_mem_megabytes),
            )
        self.logger.debug(
            f"inserted device measurements for benchmark #{benchmark_id} [CPU util: {cpu_util}, mem MBs: {mem_megabytes}, GPU util: {gpu_util}, GPU mem MBs: {gpu_mem_megabytes}]"
        )

    def collect_model_measurements(self, benchmark_id: int, measurements: Dict[str, float]):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_measurements (
                    benchmark_id,
                    measurements
                ) VALUES (%s, %s)
                """,
                (
                    benchmark_id,
                    measurements,
                ),
            )
        self.logger.debug(f"inserted model measurements for benchmark #{benchmark_id}: {measurements}")

    def close(self):
        self.conn.close()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s - %(asctime)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def parse_arguments():
    """
    Parse command line arguments for the benchmarking CLI.
    """
    parser = argparse.ArgumentParser(description="CLI for benchmarking the huggingface/transformers.")

    parser.add_argument(
        "branch",
        type=str,
        help="The branch name on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_id",
        type=str,
        help="The commit hash on which the benchmarking is performed.",
    )

    parser.add_argument(
        "commit_msg",
        type=str,
        help="The commit message associated with the commit, truncated to 70 characters.",
    )

    args = parser.parse_args()

    return args.branch, args.commit_id, args.commit_msg


def import_from_path(module_name, file_path):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        raise ImportModuleException(f"failed to load python module: {e}")


if __name__ == "__main__":
    benchmarks_folder_path = os.path.dirname(os.path.realpath(__file__))

    branch, commit_id, commit_msg = parse_arguments()

    for entry in os.scandir(benchmarks_folder_path):
        try:
            if not entry.name.endswith(".py"):
                continue
            if entry.path == __file__:
                continue
            logger.debug(f"loading: {entry.name}")
            module = import_from_path(entry.name.split(".")[0], entry.path)
            logger.info(f"runnning benchmarks in: {entry.name}")
            module.run_benchmark(logger, branch, commit_id, commit_msg)
        except ImportModuleException as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"error running benchmarks for {entry.name}: {e}")
