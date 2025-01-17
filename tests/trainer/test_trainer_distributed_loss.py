import json

import datasets
import torch

from tests.trainer.test_trainer import StoreLossCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.testing_utils import (
    TestCasePlus,
    execute_subprocess_async,
    get_torch_dist_unique_port,
    require_torch_multi_gpu,
)


class TestTrainerDistributed(TestCasePlus):
    @require_torch_multi_gpu
    def test_trainer(self):
        device_count = torch.cuda.device_count()
        min_bs = 1
        output_dir = self.get_auto_remove_tmp_dir()
        for gpu_num, enable, bs, name in (
            (1, True, min_bs * device_count, "base"),
            (device_count, False, min_bs, "broken"),
            (device_count, True, min_bs, "fixed"),
        ):
            distributed_args = f"""--nproc_per_node={gpu_num}
                --master_port={get_torch_dist_unique_port()}
                {self.test_file_dir}/test_trainer_distributed_loss.py
            """.split()
            args = f"--output_dir {output_dir}/{name} --per_device_train_batch_size {bs} --average_tokens_across_devices {enable}".split()
            cmd = ["torchrun"] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        with open(f"{output_dir}/base_losses.json") as f:
            base_loss = json.load(f)
        with open(f"{output_dir}/broken_losses.json") as f:
            broken_loss = json.load(f)
        with open(f"{output_dir}/fixed_losses.json") as f:
            fixed_loss = json.load(f)

        broken_diff = [abs(base_loss[i] - broken_loss[i]) for i in range(len(base_loss))]
        fixed_diff = [abs(base_loss[i] - fixed_loss[i]) for i in range(len(base_loss))]
        sum_base = sum(base_loss)
        sum_broken = sum(broken_diff)
        relative_broken = abs(sum_broken - sum_broken) / max(sum_base, sum_broken)

        self.assertGreater(max(broken_diff), 0.5)
        self.assertLess(max(fixed_diff), 0.005)
        self.assertLess(relative_broken, 0.1)


def run_training(training_args):
    set_seed(42)
    model_name = "nickypro/tinyllama-15M"
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    dataset = datasets.load_dataset(dataset_name, dataset_config, split="train[:17]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=16, padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    loss_callback = StoreLossCallback()

    training_args.logging_steps = 1
    training_args.max_steps = 10
    training_args.learning_rate = 3e-4
    training_args.disable_tqdm = True
    training_args.dataloader_drop_last = True
    training_args.report_to = []

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset,
        callbacks=[loss_callback],
        data_collator=data_collator,
    )
    trainer.train()
    with open(training_args.output_dir + "_losses.json", "w") as f:
        json.dump(loss_callback.losses, f)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    run_training(training_args)
