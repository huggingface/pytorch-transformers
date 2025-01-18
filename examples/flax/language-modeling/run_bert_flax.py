#!/usr/bin/env python3
import time
from argparse import ArgumentParser

import jax
import numpy as np

from transformers import BertConfig, FlaxBertModel


parser = ArgumentParser()
parser.add_argument("--precision", type=str, choices=["float32", "bfloat16"], default="float32")
args = parser.parse_args()

dtype = jax.numpy.float32
if args.precision == "bfloat16":
    dtype = jax.numpy.bfloat16

VOCAB_SIZE = 30522
BS = 32
SEQ_LEN = 128


def get_input_data(batch_size=1, seq_length=384):
    shape = (batch_size, seq_length)
    input_ids = np.random.randint(1, VOCAB_SIZE, size=shape).astype(np.int32)
    token_type_ids = np.ones(shape).astype(np.int32)
    attention_mask = np.ones(shape).astype(np.int32)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}


inputs = get_input_data(BS, SEQ_LEN)
config = BertConfig.from_pretrained("bert-base-uncased", hidden_act="gelu_new")
model = FlaxBertModel.from_pretrained("bert-base-uncased", config=config, dtype=dtype)


@jax.jit
def func():
    outputs = model(**inputs)
    return outputs


(nwarmup, nbenchmark) = (5, 100)

# warmpup
for _ in range(nwarmup):
    func()

# benchmark

start = time.time()
for _ in range(nbenchmark):
    func()
end = time.time()
print(end - start)
print(f"Throughput: {((nbenchmark * BS)/(end-start)):.3f} examples/sec")
