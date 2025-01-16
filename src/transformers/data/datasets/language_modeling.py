import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from filelock import FileLock
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer  # Ensure to import your tokenizer here

logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")
                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")

        logger.info(f"Creating features from dataset file at {file_path}")
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class LineByLineWithRefDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_wwm.py"
            ),
            FutureWarning,
        )
        if not os.path.isfile(file_path):
            raise ValueError(f"Input file path {file_path} not found")
        if not os.path.isfile(ref_path):
            raise ValueError(f"Ref file path {ref_path} not found")

        logger.info(f"Creating features from dataset file at {file_path}")
        logger.info(f"Use ref segment results at {ref_path}")
        
        with open(file_path, encoding="utf-8") as f:
            data = f.readlines()
        data = [line.strip() for line in data if len(line) > 0 and not line.isspace()]

        with open(ref_path, encoding="utf-8") as f:
            ref = [json.loads(line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        if len(data) != len(ref):
            raise ValueError(
                f"Length of Input file should be equal to Ref file. But the length of {file_path} is {len(data)} "
                f"while length of {ref_path} is {len(ref)}"
            )

        batch_encoding = tokenizer(data, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

        n = len(self.examples)
        for i in range(n):
            self.examples[i]["chinese_ref"] = torch.tensor(ref[i], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class LineByLineWithSOPTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        if not os.path.isdir(file_dir):
            raise ValueError(f"{file_dir} is not a directory")
        
        logger.info(f"Creating features from dataset file folder at {file_dir}")
        self.examples = []
        
        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir, file_name)
            if not os.path.isfile(file_path):
                raise ValueError(f"{file_path} is not a file")
            
            article_open = False
            with open(file_path, encoding="utf-8") as f:
                original_lines = f.readlines()
                article_lines = []
                for line in original_lines:
                    if "<doc id=" in line:
                        article_open = True
                    elif "</doc>" in line:
                        article_open = False
                        document = [
                            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line))
                            for line in article_lines[1:]
                            if (len(line) > 0 and not line.isspace())
                        ]

                        examples = self.create_examples_from_document(document, block_size, tokenizer)
                        self.examples.extend(examples)
                        article_lines = []
                    else:
                        if article_open:
                            article_lines.append(line)

        logger.info("Dataset parse finished.")

    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=0.1):
        max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
        target_seq_length = max_num_tokens

        if random.random() < short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)

        examples = []
        current_chunk = []
        current_length = 0
        i = 0

        while i < len(document):
            segment = document[i]
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)
            current_length += len(segment)

            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue

                    if random.random() < 0.5:
                        is_next = False
                        tokens_a, tokens_b = tokens_b, tokens_a
                    else:
                        is_next = True

                    examples.append(
                        {
                            "input_ids": tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b),
                            "is_next": is_next,
                        }
                    )
                current_chunk = []
                current_length = 0
            i += 1
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
