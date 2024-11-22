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

from typing import Dict

from transformers import is_torch_available
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_accelerate,
    require_fp8,
    require_fsdp,
    require_torch_multi_gpu,
)


if is_torch_available():
    import torch
    import torch.distributed
    import torch.utils.data

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EvalPrediction,
        GenerationConfig,
        HfArgumentParser,
        PreTrainedTokenizerBase,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    class DummyTextDataset(torch.utils.data.Dataset[str]):
        def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
            data = 4 * [
                "Hello world!",
                "The quick brown fox jumps over the lazy dog.",
            ]
            self.data = [
                {k: v.squeeze(0) for k, v in tokenizer(item, return_tensors="pt", return_attention_mask=True).items()}
                for item in data
            ]
            for item in self.data:
                item["labels"] = item["input_ids"]

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, i: int) -> str:
            return self.data[i]


class TestFSDPTrainer(TestCasePlus):
    @require_accelerate
    @require_torch_multi_gpu
    @require_fsdp
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{torch.cuda.device_count()}",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call


class TestFSDPTrainerFP8(TestCasePlus):
    @require_accelerate
    @require_torch_multi_gpu
    @require_fsdp
    @require_fp8
    def test_trainer(self):
        output_dir = self.get_auto_remove_tmp_dir()
        cmd = [
            "accelerate",
            "launch",
            "--use_fsdp",
            "--main_process_port",
            f"{get_torch_dist_unique_port()}",
            "--num_processes",
            f"{torch.cuda.device_count()}",
            "--mixed_precision",
            "fp8",
            "--fsdp_transformer_layer_cls_to_wrap",
            "GPT2Block",
            f"{self.test_file_dir}/test_trainer_fsdp.py",
            "--output_dir",
            f"{output_dir}",
            "--report_to",
            "none",
        ]
        execute_subprocess_async(cmd, env=self.get_env())
        # successful return here == success - any errors would have caused an error in the sub-call

    class TestFSDPTrainerWrap(TestCasePlus):
        @require_accelerate
        @require_torch_multi_gpu
        @require_fsdp
        def test_trainer(self):
            output_dir = self.get_auto_remove_tmp_dir()
            cmd = [
                "accelerate",
                "launch",
                "--use_fsdp",
                "--main_process_port",
                f"{get_torch_dist_unique_port()}",
                "--num_processes",
                f"{torch.cuda.device_count()}",
                "--fsdp_transformer_layer_cls_to_wrap",
                "GPT2Block",
                f"{self.test_file_dir}/test_trainer_fsdp.py",
                "--output_dir",
                f"{output_dir}",
                "--report_to",
                "none",
                "--auto_find_batch_size",
                "True",
            ]
            execute_subprocess_async(cmd, env=self.get_env())
            # successful return here == success - any errors would have caused an error in the sub-call


if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    training_args.per_device_eval_batch_size = 1
    training_args.use_legacy_prediction_loop = False
    training_args.predict_with_generate = True
    training_args.generation_config = GenerationConfig(max_length=30)

    pretrained_model_name = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(torch.distributed.get_rank())
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name).to(device)

    def compute_metrics(p: EvalPrediction) -> Dict[str, bool]:
        return {"accuracy": (p.predictions == p.label_ids).mean()}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        eval_dataset=DummyTextDataset(tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
