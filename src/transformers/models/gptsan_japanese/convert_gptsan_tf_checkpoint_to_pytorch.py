# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

"""Convert GPTSANJapanese checkpoints from the original repository to pytorch model."""

import tensorflow as tf
import torch
import numpy as np
import argparse
import json
import os
from collections import OrderedDict

parser = argparse.ArgumentParser(
    description='model converter.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--tf_model_dir', metavar='PATH', type=str, required=True, help='import model')
parser.add_argument('--output', metavar='PATH', type=str, required=True, help='output model')

def main():
    args = parser.parse_args()
    params = json.loads(open(os.path.join(args.tf_model_dir,"parameters.json")).read())
    assert params
    if not args.output.endswith(".pt"):
        args.output = args.output+".pt"

    new_state = OrderedDict()

    with tf.device("/CPU:0"):
        reader = tf.train.load_checkpoint(args.tf_model_dir)
        shapes = reader.get_variable_to_shape_map()

        for k,s in shapes.items():
            vnp = reader.get_tensor(k).astype(np.float16)
            if k.endswith("/adam_m") or k.endswith("/adam_v"):
                continue
            if k.startswith("pasts/"):
                if k.startswith("pasts/mlp"):
                    player = int(k[9])
                elif k.startswith("pasts/out"):
                    player = 8
                name = 'spout.%d.weight'%(player*2) # enter to nn.Sequencial with Tanh, so 2 at a time
                state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                new_state[name] = torch.tensor(state)
            elif k.startswith("model/moe"):
                player = int(k[9:].split("/")[0])
                if k.endswith("/switch_gating/kernel"):
                    name = 'blocks.%d.ff.mlp.router.classifier.weight'%player
                    state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/softmlp/kernel"):
                    name = 'blocks.%d.ff.smlp.weight'%player
                    state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/wo/kernel") or k.endswith("/wi/kernel"):
                    nlayer = k[-9:-7]
                    for i in range(16):
                        name = 'blocks.%d.ff.mlp.experts.expert_%d.%s.weight'%(player,i,nlayer)
                        state = vnp[i].transpose([1,0]).copy() # In Mesh-Tensorflow, it is one array, so it is divided
                        new_state[name] = torch.tensor(state)
            elif k.startswith("model/mlp"):
                player = int(k[9:].split("/")[0])
                if k.endswith("/p1/kernel"):
                    name = 'blocks.%d.ff.mlp.wi.weight'%player
                    state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/p1/bias"):
                    name = 'blocks.%d.ff.mlp.wi.bias'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/p2/kernel"):
                    name = 'blocks.%d.ff.mlp.wo.weight'%player
                    state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/p2/bias"):
                    name = 'blocks.%d.ff.mlp.wo.bias'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif k.startswith("model/ln"):
                player = int(k[8:].split("/")[0])
                if k.endswith("/b"):
                    name = 'blocks.%d.ff.norm.bias'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/g"):
                    name = 'blocks.%d.ff.norm.weight'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif k.startswith("model/att"):
                player = int(k[9:].split("/")[0])
                if k.endswith("/qkv/kernel"):
                    name = 'blocks.%d.att.SelfAttention.qkv'%player
                    state = vnp.copy() # Compute same dimension as Mesh-tensorflow using einsum
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/o/kernel"):
                    name = 'blocks.%d.att.SelfAttention.o'%player
                    state = vnp.copy() # Compute same dimension as Mesh-tensorflow using einsum
                    new_state[name] = torch.tensor(state)
            elif k.startswith("model/an"):
                player = int(k[8:].split("/")[0])
                if k.endswith("/b"):
                    name = 'blocks.%d.att.norm.bias'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif k.endswith("/g"):
                    name = 'blocks.%d.att.norm.weight'%player
                    state = vnp.copy() # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif k.startswith("model/wte") or k.startswith("model/wob") or k.startswith("model/wpe") or k.startswith("model/ete"):
                nlayer = k[-3:]
                name = '%s.weight'%nlayer if nlayer!="wob" else "wob"
                state = vnp.copy() # same in embedded
                new_state[name] = torch.tensor(state)
            elif k=="model/dense/kernel":
                name = 'logits.weight'
                state = vnp.transpose([1,0]).copy() # Mesh-Tensorflow is a diagonal matrix
                new_state[name] = torch.tensor(state)
            elif k=="model/dense_1/bias":
                name = 'logits.bias'
                state = vnp.copy() # same because it is one dimensional
                new_state[name] = torch.tensor(state)

    torch.save(new_state, args.output)

if __name__ == '__main__':
    main()
