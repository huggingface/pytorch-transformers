<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed Integration

[DeepSpeed](https://github.com/microsoft/DeepSpeed) は、[ZeRO 論文](https://arxiv.org/abs/1910.02054) で説明されているすべてを実装します。現在、次のものを完全にサポートしています。

1. オプティマイザーの状態分割 (ZeRO ステージ 1)
2. 勾配分割 (ZeRO ステージ 2)
3. パラメーターの分割 (ZeRO ステージ 3)
4. カスタム混合精度トレーニング処理
5. 一連の高速 CUDA 拡張ベースのオプティマイザー
6. CPU および NVMe への ZeRO オフロード

ZeRO-Offload には独自の専用ペーパーがあります: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)。 NVMe サポートについては、論文 [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)。

DeepSpeed ZeRO-2 は、その機能が推論には役に立たないため、主にトレーニングのみに使用されます。

DeepSpeed ZeRO-3 は、巨大なモデルを複数の GPU にロードできるため、推論にも使用できます。
単一の GPU では不可能です。

🤗 Transformers は、2 つのオプションを介して [DeepSpeed](https://github.com/microsoft/DeepSpeed) を統合します。

1. [`Trainer`] によるコア DeepSpeed 機能の統合。何でもやってくれるタイプです
   統合の場合 - カスタム構成ファイルを指定するか、テンプレートを使用するだけで、他に何もする必要はありません。たいていの
   このドキュメントではこの機能に焦点を当てています。
2. [`Trainer`] を使用せず、DeepSpeed を統合した独自のトレーナーを使用したい場合
   `from_pretrained` や `from_config` などのコア機能には、重要な機能の統合が含まれています。
   ZeRO ステージ 3 以降の `zero.Init`などの DeepSpeed の部分。この機能を活用するには、次のドキュメントをお読みください。
   [非トレーナー DeepSpeed 統合](#nontrainer-deepspeed-integration)。

統合されているもの:

トレーニング：

1. DeepSpeed ZeRO トレーニングは、ZeRO-Infinity (CPU および NVME オフロード) を使用して完全な ZeRO ステージ 1、2、および 3 をサポートします。

推論：

1. DeepSpeed ZeRO Inference は、ZeRO-Infinity による ZeRO ステージ 3 をサポートします。トレーニングと同じ ZeRO プロトコルを使用しますが、
   オプティマイザと lr スケジューラは使用せず、ステージ 3 のみが関連します。詳細については、以下を参照してください。
   [ゼロ推論](#zero-inference)。

DeepSpeed Inference もあります。これは、Tensor Parallelism の代わりに Tensor Parallelism を使用するまったく異なるテクノロジーです。
ZeRO (近日公開)。

<a id='deepspeed-trainer-integration'></a>


## Trainer Deepspeed Integration


<a id='deepspeed-installation'></a>

### Installation

pypi 経由でライブラリをインストールします。
```bash
pip install deepspeed
```

または`tansformers`, `extras`経由:

```bash
pip install transformers[deepspeed]
```

または、[DeepSpeed の GitHub ページ](https://github.com/microsoft/deepspeed#installation) で詳細を確認してください。
[高度なインストール](https://www.deepspeed.ai/tutorials/advanced-install/)。

それでもビルドに苦労する場合は、まず [CUDA 拡張機能のインストール ノート](trainer#cuda-extension-installation-notes) を必ず読んでください。

拡張機能を事前ビルドせず、実行時に拡張機能がビルドされることに依存しており、上記の解決策をすべて試した場合
それが役に立たなかった場合、次に試すべきことは、モジュールをインストールする前にモジュールを事前にビルドすることです。

DeepSpeed のローカル ビルドを作成するには:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

NVMe オフロードを使用する場合は、上記の手順に`DS_BUILD_AIO=1`を含める必要があります (また、
*libaio-dev* システム全体にインストールします)。

`TORCH_CUDA_ARCH_LIST` を編集して、使用する GPU カードのアーキテクチャのコードを挿入します。すべてを仮定すると
あなたのカードは同じで、次の方法でアーチを取得できます。

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

したがって、`8, 6`を取得した場合は、`TORCH_CUDA_ARCH_LIST="8.6"`を使用します。複数の異なるカードをお持ちの場合は、すべてをリストすることができます
それらのうち、`TORCH_CUDA_ARCH_LIST="6.1;8.6"`が好きです

複数のマシンで同じセットアップを使用する必要がある場合は、バイナリ ホイールを作成します。

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

`dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`のようなものが生成されるので、これをインストールできます
`pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`としてローカルまたは他のマシンにインストールします。

繰り返しますが、`TORCH_CUDA_ARCH_LIST`をターゲット アーキテクチャに合わせて調整することを忘れないでください。

NVIDIA GPU の完全なリストと、それに対応する **コンピューティング機能** (この記事の Arch と同じ) を見つけることができます。
コンテキスト) [ここ](https://developer.nvidia.com/cuda-gpus)。

以下を使用して、pytorch が構築されたアーチを確認できます。

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

ここでは、インストールされている GPU の 1 つのアーチを見つける方法を説明します。たとえば、GPU 0 の場合:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))"
```

出力が次の場合:

```bash
_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)
```

そうすれば、このカードのアーチが`8.6`であることがわかります。

`TORCH_CUDA_ARCH_LIST` を完全に省略することもできます。そうすれば、ビルド プログラムが自動的にクエリを実行します。
ビルドが行われる GPU のアーキテクチャ。これは、ターゲット マシンの GPU と一致する場合もあれば、一致しない場合もあります。
目的のアーチを明示的に指定することをお勧めします。

提案されたことをすべて試してもまだビルドの問題が発生する場合は、GitHub の問題に進んでください。
[ディープスピード](https://github.com/microsoft/DeepSpeed/issues)、

<a id='deepspeed-multi-gpu'></a>

### Deployment with multiple GPUs

DeepSpeed 統合をデプロイするには、[`Trainer`] コマンド ライン引数を調整して新しい引数 `--deepspeed ds_config.json` を含めます。ここで、`ds_config.json` は DeepSpeed 構成ファイルです。
   [こちら](https://www.deepspeed.ai/docs/config-json/)に記載されています。ファイル名はあなた次第です。
   DeepSpeed の`add_config_arguments`ユーティリティを使用して、必要なコマンド ライン引数をコードに追加することをお勧めします。
   詳細については、[DeepSpeed の引数解析](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) ドキュメントを参照してください。

ここで選択したランチャーを使用できます。 pytorch ランチャーを引き続き使用できます。

```bash
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

または、`deepspeed`によって提供されるランチャーを使用します。

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

ご覧のとおり、引数は同じではありませんが、ほとんどのニーズではどちらでも機能します。の
さまざまなノードと GPU を構成する方法の詳細については、[こちら](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) を参照してください。

`deepspeed`ランチャーを使用し、利用可能なすべての GPU を使用したい場合は、`--num_gpus`フラグを省略するだけです。

以下は、利用可能なすべての GPU をデプロイする DeepSpeed で`run_translation.py`を実行する例です。

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

DeepSpeed のドキュメントには、`--deepspeed --deepspeed_config ds_config.json`が表示される可能性が高いことに注意してください。
DeepSpeed 関連の引数が 2 つありますが、簡単にするためであり、処理すべき引数がすでに非常に多いためです。
この 2 つを 1 つの引数に結合しました。

実際の使用例については、この [投稿](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400) を参照してください。

<a id='deepspeed-one-gpu'></a>


### Deployment with one GPU

1 つの GPU で DeepSpeed をデプロイするには、[`Trainer`] コマンド ライン引数を次のように調整します。

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

これは複数の GPU の場合とほぼ同じですが、ここでは、DeepSpeed に 1 つの GPU だけを使用するように明示的に指示します。
`--num_gpus=1`。デフォルトでは、DeepSpeed は指定されたノード上で認識できるすべての GPU をデプロイします。起動する GPU が 1 つだけの場合
の場合、この引数は必要ありません。次の [ドキュメント](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) では、ランチャー オプションについて説明しています。

1 つの GPU だけで DeepSpeed を使用したいのはなぜですか?

1. 一部の計算とメモリをホストの CPU と RAM に委任できる ZeRO オフロード機能を備えているため、
   モデルのニーズに合わせてより多くの GPU リソースを残しておきます。より大きなバッチ サイズ、または非常に大きなモデルのフィッティングを可能にする
   普通は合わないでしょう。
2. スマートな GPU メモリ管理システムを提供し、メモリの断片化を最小限に抑えます。
   より大きなモデルとデータ バッチ。

次に構成について詳しく説明しますが、単一の GPU で大幅な改善を実現するための鍵は次のとおりです。
DeepSpeed を使用するには、構成ファイルに少なくとも次の構成が必要です。

```json
{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}
```

これにより、オプティマイザーのオフロードやその他の重要な機能が有効になります。バッファ サイズを試してみるとよいでしょう。
詳細については、以下のディスカッションを参照してください。

このタイプのデプロイメントの実際的な使用例については、この [投稿](https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685) を参照してください。

このドキュメントで詳しく説明されているように、CPU および NVMe オフロードを備えた ZeRO-3 を試すこともできます。

ノート：

- GPU 0 とは異なる特定の GPU で実行する必要がある場合、`CUDA_VISIBLE_DEVICES` を使用して制限することはできません。
  利用可能な GPU の表示範囲。代わりに、次の構文を使用する必要があります。

  ```bash
  deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
  ```

  この例では、DeepSpeed に GPU 1 (2 番目の GPU) を使用するように指示します。

<a id='deepspeed-multi-node'></a>

### 複数のノードを使用したデプロイメント

このセクションの情報は DeepSpeed 統合に固有のものではなく、あらゆるマルチノード プログラムに適用できます。ただし、DeepSpeed は、SLURM 環境でない限り、他のランチャーよりも使いやすい`deepspeed`ランチャーを提供します。

このセクションでは、それぞれ 8 GPU を備えた 2 つのノードがあると仮定します。また、最初のノードには `ssh hostname1` を使用して、2 番目のノードには `ssh hostname2` を使用して接続できます。両方ともパスワードなしでローカルの ssh 経由で相互に接続できる必要があります。もちろん、これらのホスト (ノード) 名を、作業している実際のホスト名に変更する必要があります。

#### The torch.distributed.run launcher


たとえば、`torch.distributed.run` を使用するには、次のようにします。

```bash
python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

各ノードに SSH で接続し、それぞれのノードで同じコマンドを実行する必要があります。急ぐ必要はありません。ランチャーは両方のノードが同期するまで待機します。

詳細については、[torchrun](https://pytorch.org/docs/stable/elastic/run.html) を参照してください。ちなみに、これは pytorch の数バージョン前の`torch.distributed.launch`を置き換えたランチャーでもあります。

#### ディープスピード ランチャー

代わりに`deepspeed`ランチャーを使用するには、まず`hostfile`ファイルを作成する必要があります。

```
hostname1 slots=8
hostname2 slots=8
```

そして、次のように起動できます。

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

`torch.distributed.run`ランチャーとは異なり、`deepspeed`は両方のノードでこのコマンドを自動的に起動します。

詳細については、[リソース構成 (マルチノード)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) を参照してください。

#### Launching in a SLURM environment

SLURM 環境では、次のアプローチを使用できます。以下は、特定の SLURM 環境に適合させるために必要な slurm スクリプト `launch.slurm` です。

```bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

あとは実行をスケジュールするだけです。
```bash
sbatch launch.slurm
```

#### Use of Non-shared filesystem

デフォルトでは、DeepSpeed はマルチノード環境が共有ストレージを使用することを想定しています。これが当てはまらず、各ノードがローカル ファイルシステムしか参照できない場合は、設定ファイルを調整して [`checkpoint`_section](https://www.deepspeed.ai/docs/config-json/#) を含める必要があります。チェックポイント オプション) を次の設定で指定します。


```json
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

あるいは、[`Trainer`] の `--save_on_each_node` 引数を使用することもでき、上記の設定は自動的に追加されます。

<a id='deepspeed-notebook'></a>

### Deployment in Notebooks

ノートブックのセルをスクリプトとして実行する場合の問題は、依存する通常の`deepspeed`ランチャーがないことです。
特定の設定では、それをエミュレートする必要があります。

GPU を 1 つだけ使用している場合、DeepSpeed を使用するためにノートブック内のトレーニング コードを調整する必要がある方法は次のとおりです。

```python
# DeepSpeed requires a distributed environment even when only one process is used.
# This emulates a launcher in the notebook
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Now proceed as normal, plus pass the deepspeed config file
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

注: `...` は、関数に渡す通常の引数を表します。

複数の GPU を使用する場合、DeepSpeed が動作するにはマルチプロセス環境を使用する必要があります。つまり、あなたは持っています
その目的でランチャーを使用することはできませんが、これは、提示された分散環境をエミュレートすることによっては実現できません。
このセクションの冒頭で。

現在のディレクトリのノートブックにその場で構成ファイルを作成したい場合は、専用の
セルの内容:

```python no-style
%%bash
cat <<'EOT' > ds_config_zero3.json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOT
```

トレーニング スクリプトがノートブックのセルではなく通常のファイルにある場合は、次のようにして`deepspeed`を通常どおり起動できます。
細胞からのシェル。たとえば、`run_translation.py` を使用するには、次のように起動します。

```python no-style
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

または、`%%bash` マジックを使用すると、シェル プログラムを実行するための複数行のコードを記述することができます。

```python no-style
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

そのような場合、このセクションの最初に示したコードは必要ありません。

注: `%%bash` マジックは優れていますが、現時点では出力をバッファリングするため、プロセスが終了するまでログは表示されません。
完了します。

<a id='deepspeed-config'></a>

### Configuration

設定ファイルで使用できる DeepSpeed 設定オプションの完全なガイドについては、次を参照してください。
[次のドキュメント](https://www.deepspeed.ai/docs/config-json/) にアクセスしてください。

さまざまな実際のニーズに対応する数十の DeepSpeed 構成例を [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)で見つけることができます。
リポジトリ:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
```

上記のコードを続けて、Lamb オプティマイザーを構成しようとしているとします。したがって、次の中から検索できます
`.json` ファイルの例:

```bash
grep -i Lamb $(find . -name '*json')
```

さらにいくつかの例が [メイン リポジトリ](https://github.com/microsoft/DeepSpeed) にもあります。

DeepSpeed を使用する場合は、常に DeepSpeed 構成ファイルを指定する必要がありますが、一部の構成パラメータには
コマンドライン経由で設定します。微妙な違いについては、このガイドの残りの部分で説明します。

DeepSpeed 構成ファイルがどのようなものかを理解するために、ZeRO ステージ 2 機能を有効にする構成ファイルを次に示します。
オプティマイザー状態の CPU オフロードを含み、`AdamW`オプティマイザーと`WarmupLR`スケジューラーを使用し、混合を有効にします。
`--fp16` が渡された場合の精度トレーニング:


```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
```

プログラムを実行すると、DeepSpeed は [`Trainer`] から受け取った設定をログに記録します。
コンソールに渡されるため、最終的にどのような設定が渡されたのかを正確に確認できます。

<a id='deepspeed-config-passing'></a>

### Passing Configuration

このドキュメントで説明したように、通常、DeepSpeed 設定は json ファイルへのパスとして渡されますが、
トレーニングの設定にコマンド ライン インターフェイスを使用せず、代わりにインスタンスを作成します。
[`Trainer`] via [`TrainingArguments`] その後、`deepspeed` 引数については次のことができます
ネストされた `dict` を渡します。これにより、その場で構成を作成でき、それを書き込む必要がありません。
[`TrainingArguments`] に渡す前にファイル システムを変更します。

要約すると、次のことができます。

```python
TrainingArguments(..., deepspeed="/path/to/ds_config.json")
```

または：

```python
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
TrainingArguments(..., deepspeed=ds_config_dict)
```

<a id='deepspeed-config-shared'></a>


### Shared Configuration

<Tip warning={true}>

このセクションは必読です

</Tip>

[`Trainer`] と DeepSpeed の両方が正しく機能するには、いくつかの設定値が必要です。
したがって、検出が困難なエラーにつながる可能性のある定義の競合を防ぐために、それらを構成することにしました。
[`Trainer`] コマンドライン引数経由。

さらに、一部の構成値はモデルの構成に基づいて自動的に導出されます。
複数の値を手動で調整することを忘れないでください。[`Trainer`] に大部分を任せるのが最善です
の設定を行います。

したがって、このガイドの残りの部分では、特別な設定値 `auto` が表示されます。これを設定すると、
正しい値または最も効率的な値に自動的に置き換えられます。これを無視することを自由に選択してください
推奨事項を参照し、値を明示的に設定します。この場合、次の点に十分注意してください。
[`Trainer`] 引数と DeepSpeed 設定は一致します。たとえば、同じものを使用していますか
学習率、バッチサイズ、または勾配累積設定?これらが一致しない場合、トレーニングは非常に失敗する可能性があります
方法を検出するのが難しい。あなたは警告を受けました。

DeepSpeed のみに固有の値や、それに合わせて手動で設定する必要がある値が他にも複数あります。
あなたの要望。

独自のプログラムで、DeepSpeed 構成をマスターとして変更したい場合は、次のアプローチを使用することもできます。
それに基づいて [`TrainingArguments`] を設定します。手順は次のとおりです。

1. マスター構成として使用する DeepSpeed 構成を作成またはロードします
2. これらの値に基づいて [`TrainingArguments`] オブジェクトを作成します

`scheduler.params.total_num_steps`などの一部の値は次のように計算されることに注意してください。
`train` 中に [`Trainer`] を実行しますが、もちろん自分で計算することもできます。

<a id='deepspeed-zero'></a>

### ZeRO

[Zero Redundancy Optimizer (ZeRO)](https://www.deepspeed.ai/tutorials/zero/) は、DeepSpeed の主力製品です。それ
3 つの異なるレベル (段階) の最適化をサポートします。最初のものは、スケーラビリティの観点からはあまり興味深いものではありません。
したがって、このドキュメントではステージ 2 と 3 に焦点を当てます。ステージ 3 は、最新の ZeRO-Infinity の追加によってさらに改善されています。
詳細については、DeepSpeed のドキュメントを参照してください。

構成ファイルの `zero_optimization` セクションは最も重要な部分です ([docs](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training))。ここで定義します
どの ZeRO ステージを有効にするか、そしてそれらをどのように構成するか。各パラメータの説明は、
DeepSpeed のドキュメント。

このセクションは、DeepSpeed 設定を介してのみ設定する必要があります - [`Trainer`] が提供します
同等のコマンドライン引数はありません。

注: 現在、DeepSpeed はパラメーター名を検証しないため、スペルを間違えると、デフォルト設定が使用されます。
スペルが間違っているパラメータ。 DeepSpeed エンジンの起動ログ メッセージを見て、その値を確認できます。
使用するつもりです。

<a id='deepspeed-zero2-config'></a>

#### ZeRO-2 Config

以下は、ZeRO ステージ 2 の構成例です。

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    }
}
```

**性能調整：**

- `offload_optimizer` を有効にすると、GPU RAM の使用量が削減されます (`"stage": 2` が必要です)
- `"overlap_comm": true` は、GPU RAM 使用量の増加とトレードオフして、遅延をすべて削減します。 `overlap_comm`は 4.5x を使用します
  `allgather_bucket_size`と`reduce_bucket_size`の値。したがって、5e8 に設定されている場合、9GB が必要になります。
  フットプリント (`5e8 x 2Bytes x 2 x 4.5`)。したがって、8GB 以下の RAM を搭載した GPU を使用している場合、
  OOM エラーが発生した場合は、これらのパラメータを`2e8`程度に減らす必要があり、それには 3.6GB が必要になります。やりたくなるでしょう
  OOM に達し始めている場合は、より大容量の GPU でも同様です。
- これらのバッファを減らすと、より多くの GPU RAM を利用するために通信速度を犠牲にすることになります。バッファサイズが小さいほど、
  通信が遅くなり、他のタスクで使用できる GPU RAM が増えます。したがって、バッチサイズが大きい場合は、
  重要なのは、トレーニング時間を少し遅らせることは良いトレードになる可能性があります。

さらに、`deepspeed==0.4.4`には、次のコマンドで有効にできる新しいオプション`round_robin_gradients`が追加されました。

```json
{
    "zero_optimization": {
        "round_robin_gradients": true
    }
}
```

これは、きめ細かい勾配パーティショニングによってランク間の CPU メモリへの勾配コピーを並列化する、CPU オフロードのステージ 2 最適化です。パフォーマンスの利点は、勾配累積ステップ (オプティマイザー ステップ間のコピーの増加) または GPU 数 (並列処理の増加) に応じて増加します。

<a id='deepspeed-zero3-config'></a>

#### ZeRO-3 Config

以下は、ZeRO ステージ 3 の構成例です。

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

モデルまたはアクティベーションが GPU メモリに適合せず、CPU が未使用であるために OOM が発生している場合
`"device": "cpu"` を使用してオプティマイザの状態とパラメータを CPU メモリにメモリオフロードすると、この制限が解決される可能性があります。
CPU メモリにオフロードしたくない場合は、`device`エントリに`cpu`の代わりに`none`を使用します。オフロード先
NVMe については後ほど説明します。

固定メモリは、`pin_memory`を`true`に設定すると有効になります。この機能により、次のようなコストをかけてスループットを向上させることができます。
他のプロセスが使用できるメモリが少なくなります。ピン留めされたメモリは、それを要求した特定のプロセスのために確保されます。
通常、通常の CPU メモリよりもはるかに高速にアクセスされます。

**性能調整：**

- `stage3_max_live_parameters`: `1e9`
- `stage3_max_reuse_distance`: `1e9`

OOM に達した場合は、「stage3_max_live_parameters」と「stage3_max_reuse_ distance」を減らします。影響は最小限に抑えられるはずです
アクティブ化チェックポイントを実行しない限り、パフォーマンスに影響します。 `1e9`は約 2GB を消費します。記憶を共有しているのは、
`stage3_max_live_parameters` と `stage3_max_reuse_distance` なので、加算されるものではなく、合計で 2GB になります。

`stage3_max_live_parameters` は、特定の時点で GPU 上に保持する完全なパラメータの数の上限です。
時間。 「再利用距離」は、パラメータが将来いつ再び使用されるかを判断するために使用する指標です。
`stage3_max_reuse_ distance`を使用して、パラメータを破棄するか保持するかを決定します。パラメータが
近い将来に再び使用される予定 (`stage3_max_reuse_distance`未満) なので、通信を減らすために保持します。
オーバーヘッド。これは、アクティベーション チェックポイントを有効にしている場合に非常に役立ちます。フォワード再計算が行われ、
backward は単一レイヤー粒度を渡し、後方再計算までパラメータを前方再計算に保持したいと考えています。

次の構成値は、モデルの非表示サイズによって異なります。

- `reduce_bucket_size`: `hidden_size*hidden_size`
- `stage3_prefetch_bucket_size`: `0.9 * hidden_size * hidden_size`
- `stage3_param_persistence_threshold`: `10 * hidden_size`

したがって、これらの値を `auto` に設定すると、[`Trainer`] が推奨される値を自動的に割り当てます。
価値観。ただし、もちろん、これらを明示的に設定することもできます。

`stage3_gather_16bit_weights_on_model_save` は、モデルの保存時にモデル fp16 の重み統合を有効にします。大きい
モデルと複数の GPU の場合、これはメモリと速度の両方の点で高価な操作です。現在必須となっているのは、
トレーニングを再開する予定です。この制限を取り除き、より便利にする今後のアップデートに注目してください。
フレキシブル。

ZeRO-2 構成から移行している場合は、`allgather_partitions`、`allgather_bucket_size`、および
`reduce_scatter`設定パラメータは ZeRO-3 では使用されません。これらを設定ファイルに保存しておくと、
無視される。

- `sub_group_size`: `1e9`


`sub_group_size` は、オプティマイザーのステップ中にパラメーターが更新される粒度を制御します。パラメータは次のとおりです。
`sub_group_size` のバケットにグループ化され、各バケットは一度に 1 つずつ更新されます。 NVMeオフロードで使用する場合
したがって、ZeRO-Infinity の `sub_group_size`は、モデルの状態が CPU に出入りする粒度を制御します。
オプティマイザステップ中に NVMe からメモリを取得します。これにより、非常に大規模なモデルの CPU メモリ不足が防止されます。

NVMe オフロードを使用しない場合は、`sub_group_size`をデフォルト値の *1e9* のままにすることができます。変更することもできます
次の場合のデフォルト値:

1. オプティマイザー ステップ中に OOM が発生する: `sub_group_size` を減らして、一時バッファーのメモリ使用量を削減します。
2. オプティマイザー ステップに時間がかかります。`sub_group_size`を増やして、帯域幅の使用率を向上させます。
   データバッファの増加。

#### ZeRO-0 Config

ステージ 0 と 1 はめったに使用されないため、最後にリストしていることに注意してください。

ステージ 0 では、すべてのタイプのシャーディングを無効にし、DDP として DeepSpeed のみを使用します。次のコマンドでオンにできます。

```json
{
    "zero_optimization": {
        "stage": 0
    }
}
```

これにより、他に何も変更する必要がなく、基本的に ZeRO が無効になります。

#### ZeRO-1 Config

ステージ 1 は、ステージ 2 からグラデーション シャーディングを除いたものです。オプティマイザーの状態をシャード化するだけで、処理を少し高速化するためにいつでも試すことができます。

```json
{
    "zero_optimization": {
        "stage": 1
    }
}
```

<a id='deepspeed-nvme'></a>

### NVMe Support

ZeRO-Infinity は、GPU と CPU メモリを NVMe メモリで拡張することで、非常に大規模なモデルのトレーニングを可能にします。おかげで
スマート パーティショニングおよびタイリング アルゴリズムでは、各 GPU が非常に少量のデータを送受信する必要があります。
オフロードにより、最新の NVMe がトレーニングに利用できる合計メモリ プールをさらに大きくするのに適していることが判明しました。
プロセス。 ZeRO-Infinity には、ZeRO-3 が有効になっている必要があります。

次の設定例では、NVMe がオプティマイザの状態とパラメータの両方をオフロードできるようにします。

```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "fast_init": false
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        },
        "aio": {
            "block_size": 262144,
            "queue_depth": 32,
            "thread_count": 1,
            "single_submit": false,
            "overlap_events": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
}
```

オプティマイザの状態とパラメータの両方を NVMe にオフロードするか、どちらか 1 つだけをオフロードするか、まったくオフロードしないかを選択できます。たとえば、次の場合
利用可能な CPU メモリが大量にある場合は、高速になるため、必ず CPU メモリのみにオフロードしてください (ヒント:
*"device": "CPU"*)。

[オプティマイザーの状態](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) と [パラメーター](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)。

`nvme_path`が実際に NVMe であることを確認してください。NVMe は通常のハードドライブまたは SSD で動作しますが、
はるかに遅くなります。高速スケーラブルなトレーニングは、最新の NVMe 転送速度を念頭に置いて設計されました (この時点では
書き込みでは、読み取り最大 3.5 GB/秒、書き込み最大 3 GB/秒のピーク速度が得られます)。

最適な`aio`構成ブロックを見つけるには、ターゲット設定でベンチマークを実行する必要があります。
[ここで説明](https://github.com/microsoft/DeepSpeed/issues/998)。

<a id='deepspeed-zero2-zero3-performance'></a>


#### ZeRO-2 vs ZeRO-3 Performance

ZeRO-3 は、他のすべてが同じように構成されている場合、ZeRO-2 よりも遅くなる可能性があります。前者は収集する必要があるためです。
ZeRO-2 の機能に加えてモデルの重み付けを行います。 ZeRO-2 がニーズを満たし、数個の GPU を超えて拡張する必要がない場合
そうすれば、それに固執することを選択することもできます。 ZeRO-3 により、はるかに高いスケーラビリティ容量が可能になることを理解することが重要です
スピードを犠牲にして。

ZeRO-3 の構成を調整して、ZeRO-2 に近づけることができます。

- `stage3_param_persistence_threshold` を非常に大きな数値に設定します。たとえば、`6 * hidden_​​size * hidden_​​size` のように、最大​​パラメータよりも大きくなります。これにより、パラメータが GPU に保持されます。
- ZeRO-2 にはそのオプションがないため、`offload_params` をオフにします。

変更しなくても、`offload_params`をオフにするだけでパフォーマンスが大幅に向上する可能性があります。
`stage3_param_persistence_threshold`。もちろん、これらの変更はトレーニングできるモデルのサイズに影響します。それで
これらは、ニーズに応じて、スケーラビリティと引き換えに速度を向上させるのに役立ちます。

<a id='deepspeed-zero2-example'></a>

#### ZeRO-2 Example

以下は、完全な ZeRO-2 自動構成ファイル `ds_config_zero2.json` です。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

以下は、手動で設定された完全な ZeRO-2 のすべてが有効な構成ファイルです。ここでは主に、典型的なものを確認するためのものです。
値は次のようになりますが、複数の`auto`設定が含まれる値を使用することを強くお勧めします。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

<a id='deepspeed-zero3-example'></a>

#### ZeRO-3 Example

以下は、完全な ZeRO-3 自動構成ファイル`ds_config_zero3.json`です。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

以下は、手動で設定された完全な ZeRO-3 のすべてが有効な構成ファイルです。ここでは主に、典型的なものを確認するためのものです。
値は次のようになりますが、複数の`auto`設定が含まれる値を使用することを強くお勧めします。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.8, 0.999],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 1e6,
        "stage3_prefetch_bucket_size": 0.94e6,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
```

#### How to Choose Which ZeRO Stage and Offloads To Use For Best Performance

これで、さまざまな段階があることがわかりました。どちらを使用するかをどのように決定すればよいでしょうか?このセクションでは、この質問に答えていきます。

一般に、次のことが当てはまります。

- 速度の点（左の方が右より速い）

ステージ 0 (DDP) > ステージ 1 > ステージ 2 > ステージ 2 + オフロード > ステージ 3 > ステージ 3 + オフロード

- GPU メモリの使用状況 (右は左よりも GPU メモリ効率が高い)

ステージ 0 (DDP) < ステージ 1 < ステージ 2 < ステージ 2 + オフロード < ステージ 3 < ステージ 3 + オフロード

したがって、最小限の数の GPU に収まりながら最速の実行を実現したい場合は、次のプロセスに従うことができます。最も速いアプローチから開始し、GPU OOM に陥った場合は、次に遅いアプローチに進みますが、これにより使用される GPU メモリが少なくなります。などなど。

まず、バッチ サイズを 1 に設定します (必要な有効バッチ サイズに対して、いつでも勾配累積を使用できます)。

1. `--gradient_checkpointing 1` (HF Trainer) または直接 `model.gradient_checkpointing_enable()` を有効にします - OOM の場合
2. 最初に ZeRO ステージ 2 を試してください。 OOMの場合
3. ZeRO ステージ 2 + `offload_optimizer` を試します - OOM の場合
4. ZeRO ステージ 3 に切り替える - OOM の場合
5. `cpu` に対して `offload_param` を有効にします - OOM の場合
6. OOM の場合は、`cpu`に対して`offload_optimizer`を有効にします。

7. それでもバッチ サイズ 1 に適合しない場合は、まずさまざまなデフォルト値を確認し、可能であれば値を下げます。たとえば、`generate`を使用し、広い検索ビームを使用しない場合は、大量のメモリを消費するため、検索ビームを狭くします。

8. fp32 では必ず混合半精度を使用します。つまり、Ampere 以上の GPU では bf16、古い GPU アーキテクチャでは fp16 を使用します。

9. それでも OOM を行う場合は、ハードウェアを追加するか、ZeRO-Infinity を有効にすることができます。つまり、オフロード `offload_param` と `offload_optimizer` を `nvme` に切り替えます。非常に高速な nvme であることを確認する必要があります。逸話として、ZeRO-Infinity を使用して小さな GPU で BLOOM-176B を推論することができましたが、非常に遅かったです。でも、うまくいきました！

もちろん、最も GPU メモリ効率の高い構成から始めて、後から逆に進むことで、これらの手順を逆に実行することもできます。あるいは二等分してみてください。

OOM を引き起こさないバッチ サイズ 1 を取得したら、実効スループットを測定します。

次に、バッチ サイズをできるだけ大きくしてみます。バッチ サイズが大きいほど、乗算する行列が巨大な場合に GPU のパフォーマンスが最高になるため、GPU の効率が向上します。

ここで、パフォーマンス最適化ゲームが始まります。一部のオフロード機能をオフにするか、ZeRO 段階でステップダウンしてバッチ サイズを増減して、実効スループットを再度測定することができます。満足するまで洗い流し、繰り返します。

永遠にこれに費やす必要はありませんが、3 か月のトレーニングを開始しようとしている場合は、スループットに関して最も効果的な設定を見つけるために数日かけてください。そのため、トレーニングのコストが最小限になり、トレーニングをより早く完了できます。現在の目まぐるしく変化する ML の世界では、何かをトレーニングするのにさらに 1 か月かかる場合、絶好の機会を逃す可能性があります。もちろん、これは私が意見を共有しているだけであり、決してあなたを急かそうとしているわけではありません。 BLOOM-176B のトレーニングを開始する前に、このプロセスに 2 日間費やし、スループットを 90 TFLOP から 150 TFLOP に向上させることができました。この取り組みにより、トレーニング時間を 1 か月以上節約できました。

これらのメモは主にトレーニング モード用に書かれたものですが、ほとんどの場合は推論にも適用されるはずです。たとえば、勾配チェックポイントはトレーニング中にのみ役立つため、推論中は何も行われません。さらに、マルチ GPU 推論を実行していて、[DeepSpeed-Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/)、[Accelerate](https://ハグフェイス.co/blog/bloom-inference-pytorch-scripts) は優れたパフォーマンスを提供するはずです。


その他のパフォーマンス関連の簡単なメモ:
- 何かを最初からトレーニングしている場合は、常に 16 で割り切れる形状のテンソル (隠れたサイズなど) を使用するようにしてください。バッチ サイズについては、少なくとも 2 で割り切れるようにしてください。 GPU からさらに高いパフォーマンスを引き出したい場合は、ハードウェア固有の [波とタイルの量子化](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/) の可分性があります。

### Activation Checkpointing or Gradient Checkpointing

アクティベーション チェックポイントと勾配チェックポイントは、同じ方法論を指す 2 つの異なる用語です。とてもややこしいですが、こんな感じです。

勾配チェックポイントを使用すると、速度を GPU メモリと引き換えにできます。これにより、GPU OOM を克服したり、バッチ サイズを増やすことができ、多くの場合、パフォーマンスの向上につながります。

HF Transformers モデルは、DeepSpeed のアクティベーション チェックポイントについて何も知らないため、DeepSpeed 構成ファイルでその機能を有効にしようとしても、何も起こりません。

したがって、この非常に有益な機能を活用するには 2 つの方法があります。

1. HF Transformers モデルを使用したい場合は、`model.gradient_checkpointing_enable()` を実行するか、HF トレーナーで `--gradient_checkpointing` を使用します。これにより、これが自動的に有効になります。そこで使われるのが `torch.utils.checkpoint` です。
2. 独自のモデルを作成し、DeepSpeed のアクティベーション チェックポイントを使用したい場合は、[そこで規定されている API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html) を使用できます。 HF Transformers モデリング コードを使用して、`torch.utils.checkpoint` を DeepSpeed の API に置き換えることもできます。後者は、順方向アクティベーションを再計算する代わりに CPU メモリにオフロードできるため、より柔軟です。

### Optimizer and Scheduler

`offload_optimizer`を有効にしない限り、DeepSpeed スケジューラーと HuggingFace スケジューラーを組み合わせて使用​​できます。
オプティマイザー (HuggingFace スケジューラーと DeepSpeed オプティマイザーの組み合わせを除く):

| Combos       | HF Scheduler | DS Scheduler |
|:-------------|:-------------|:-------------|
| HF Optimizer | Yes          | Yes          |
| DS Optimizer | No           | Yes          |

`offload_optimizer`が有効な場合、CPU と
GPU 実装 (LAMB を除く)。


<a id='deepspeed-optimizer'></a>

#### Optimizer

DeepSpeed の主なオプティマイザーは、Adam、AdamW、OneBitAdam、Lamb です。これらは ZeRO で徹底的にテストされており、
したがって、使用することをお勧めします。ただし、他のオプティマイザを「torch」からインポートすることはできます。完全なドキュメントは [こちら](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) にあります。

設定ファイルで `optimizer` エントリを設定しない場合、[`Trainer`] は
自動的に`AdamW`に設定され、指定された値または次のコマンドラインのデフォルトが使用されます。
引数: `--learning_rate`、`--adam_beta1`、`--adam_beta2`、`--adam_epsilon`、および `--weight_decay`。

以下は、`AdamW`の自動構成された`optimizer`エントリの例です。

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": "auto",
         "betas": "auto",
         "eps": "auto",
         "weight_decay": "auto"
       }
   }
}
```

コマンドライン引数によって構成ファイル内の値が設定されることに注意してください。これは 1 つあるためです
値の決定的なソースを提供し、たとえば学習率が次のように設定されている場合に、見つけにくいエラーを回避します。
さまざまな場所でさまざまな価値観。コマンドラインのルール。オーバーライドされる値は次のとおりです。

- `lr` と `--learning_rate` の値
- `betas` と `--adam_beta1 --adam_beta2` の値
- `eps` と `--adam_epsilon` の値
- `weight_decay` と `--weight_decay` の値

したがって、コマンドラインで共有ハイパーパラメータを調整することを忘れないでください。

値を明示的に設定することもできます。

```json
{
   "optimizer": {
       "type": "AdamW",
       "params": {
         "lr": 0.001,
         "betas": [0.8, 0.999],
         "eps": 1e-8,
         "weight_decay": 3e-7
       }
   }
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

上記にリストされていない別のオプティマイザーを使用する場合は、トップレベルの構成に追加する必要があります。

```json
{
   "zero_allow_untested_optimizer": true
}
```

`AdamW`と同様に、公式にサポートされている他のオプティマイザーを構成できます。これらは異なる設定値を持つ可能性があることに注意してください。例えばAdam の場合は、`weight_decay`を`0.01`付近にする必要があります。

さらに、オフロードは、Deepspeed の CPU Adam オプティマイザーと併用すると最も効果的に機能します。 `deepspeed==0.8.3` なので、オフロードで別のオプティマイザーを使用したい場合は、以下も追加する必要があります。

```json
{
   "zero_force_ds_cpu_optimizer": false
}
```

最上位の構成に移行します。

<a id='deepspeed-scheduler'></a>


#### Scheduler


DeepSpeed は、`LRRangeTest`、`OneCycle`、`WarmupLR`、および`WarmupDecayLR`学習率スケジューラーをサポートしています。完全な
ドキュメントは[ここ](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)です。

ここでは、🤗 Transformers と DeepSpeed の間でスケジューラーが重複する場所を示します。

- `--lr_scheduler_type constant_with_warmup` 経由の `WarmupLR`
- `--lr_scheduler_type Linear` を介した `WarmupDecayLR`。これは `--lr_scheduler_type` のデフォルト値でもあります。
  したがって、スケジューラを設定しない場合、これがデフォルトで設定されるスケジューラになります。

設定ファイルで `scheduler` エントリを設定しない場合、[`Trainer`] は
`--lr_scheduler_type`、`--learning_rate`、および `--warmup_steps` または `--warmup_ratio` の値を設定します。
🤗 それのトランスフォーマーバージョン。

以下は、`WarmupLR`の自動構成された`scheduler`エントリの例です。

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

*"auto"* が使用されているため、[`Trainer`] 引数は設定に正しい値を設定します。
ファイル。これは、値の決定的なソースが 1 つあることと、たとえば次のような場合に見つけにくいエラーを避けるためです。
学習率は、場所ごとに異なる値に設定されます。コマンドラインのルール。設定される値は次のとおりです。

- `warmup_min_lr` の値は `0` です。
- `warmup_max_lr` と `--learning_rate` の値。
- `warmup_num_steps` と `--warmup_steps` の値 (指定されている場合)。それ以外の場合は `--warmup_ratio` を使用します
  トレーニング ステップの数を乗算し、切り上げます。
- `total_num_steps` には `--max_steps` の値を指定するか、指定されていない場合は実行時に自動的に導出されます。
  環境、データセットのサイズ、およびその他のコマンド ライン引数 (
  `WarmupDecayLR`)。

もちろん、構成値の一部またはすべてを引き継いで、自分で設定することもできます。

```json
{
   "scheduler": {
         "type": "WarmupLR",
         "params": {
             "warmup_min_lr": 0,
             "warmup_max_lr": 0.001,
             "warmup_num_steps": 1000
         }
     }
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

たとえば、`WarmupDecayLR`の場合は、次のエントリを使用できます。

```json
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "last_batch_iteration": -1,
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

`total_num_steps`、`warmup_max_lr`、`warmup_num_steps`、および `total_num_steps` はロード時に設定されます。

<a id='deepspeed-fp32'></a>

### fp32 Precision

Deepspeed は、完全な fp32 と fp16 の混合精度をサポートします。

fp16 混合精度を使用すると、必要なメモリが大幅に削減され、速度が向上するため、
使用しているモデルがこのトレーニング モードで適切に動作しない場合は、使用しない方がよいでしょう。通常これ
モデルが fp16 混合精度で事前トレーニングされていない場合に発生します (たとえば、これは bf16 で事前トレーニングされた場合によく発生します)
モデル）。このようなモデルでは、オーバーフローまたはアンダーフローが発生し、`NaN`損失が発生する可能性があります。これがあなたの場合は、使用したいと思うでしょう
完全な fp32 モード。デフォルトの fp16 混合精度モードを次のように明示的に無効にします。

```json
{
    "fp16": {
        "enabled": false,
    }
}
```

Ampere アーキテクチャ ベースの GPU を使用している場合、pytorch バージョン 1.7 以降は自動的に を使用するように切り替わります。
一部の操作でははるかに効率的な tf32 形式を使用しますが、結果は依然として fp32 になります。詳細と
ベンチマークについては、[Ampere デバイス上の TensorFloat-32(TF32)](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) を参照してください。文書には以下が含まれます
何らかの理由でこの自動変換を使用したくない場合は、この自動変換を無効にする方法について説明します。

🤗 トレーナーでは、`--tf32` を使用して有効にするか、`--tf32 0` または `--no_tf32` を使用して無効にすることができます。デフォルトでは、PyTorch のデフォルトが使用されます。

<a id='deepspeed-amp'></a>

### Automatic Mixed Precision

pytorch のような AMP の方法または apex のような方法で自動混合精度を使用できます。

### fp16

fp16 (float16) を設定して pytorch AMP のようなモードを設定するには:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

[`Trainer`] は、の値に基づいてそれを自動的に有効または無効にします。
`args.fp16_backend`。残りの設定値はあなた次第です。

このモードは、`--fp16 --fp16_backend amp`または`--fp16_full_eval`コマンドライン引数が渡されると有効になります。

このモードを明示的に有効/無効にすることもできます。

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

これが[ドキュメント](https://www.deepspeed.ai/docs/config-json/#fp16-training-options)です。

### BF16

fp16 の代わりに bf16 (bfloat16) が必要な場合は、次の構成セクションが使用されます。

```json
{
    "bf16": {
        "enabled": "auto"
    }
}
```

bf16 は fp32 と同じダイナミック レンジを備えているため、損失スケーリングは必要ありません。

このモードは、`--bf16` または `--bf16_full_eval` コマンドライン引数が渡されると有効になります。

このモードを明示的に有効/無効にすることもできます。

```json
{
    "bf16": {
        "enabled": true
    }
}
```

<Tip>

`deepspeed==0.6.0`の時点では、bf16 サポートは新しく実験的なものです。

bf16 が有効な状態で [勾配累積](#gradient-accumulation) を使用する場合は、bf16 で勾配が累積されることに注意する必要があります。この形式の精度が低いため、これは希望どおりではない可能性があります。損失のある蓄積につながります。

この問題を修正し、より高精度の `dtype` (fp16 または fp32) を使用するオプションを提供するための作業が行われています。

</Tip>


### NCCL Collectives

訓練体制の`dtype`があり、さまざまな削減や収集/分散操作などのコミュニケーション集合体に使用される別の`dtype`があります。

すべての収集/分散操作は、データが含まれているのと同じ `dtype` で実行されるため、bf16 トレーニング体制を使用している場合、データは bf16 で収集されます。収集は損失のない操作です。

さまざまなリデュース操作は非常に損失が大きい可能性があります。たとえば、複数の GPU 間で勾配が平均化される場合、通信が fp16 または bf16 で行われる場合、結果は損失が多くなる可能性があります。複数の数値を低精度でアドバタイズすると結果は正確ではないためです。 。 bf16 では fp16 よりも精度が低いため、さらにそうです。通常は非常に小さい grad を平均する際の損失が最小限に抑えられるため、fp16 で十分であることがよくあります。したがって、デフォルトでは、半精度トレーニングでは fp16 がリダクション演算のデフォルトとして使用されます。ただし、この機能を完全に制御でき、必要に応じて小さなオーバーヘッドを追加して、リダクションが累積 dtype として fp32 を使用し、結果の準備ができた場合にのみ半精度 `dtype` にダウンキャストするようにすることもできます。でトレーニング中です。

デフォルトをオーバーライドするには、新しい構成エントリを追加するだけです。

```json
{
    "communication_data_type": "fp32"
}
```

この記事の執筆時点での有効な値は、"fp16"、"bfp16"、"fp32"です。

注: ステージ ゼロ 3 には、bf16 通信タイプに関するバグがあり、`deepspeed==0.8.1`で修正されました。

### apex

apex AMP のようなモード セットを設定するには:

```json
"amp": {
    "enabled": "auto",
    "opt_level": "auto"
}
```

[`Trainer`] は `args.fp16_backend` の値に基づいて自動的に設定します。
`args.fp16_opt_level`。

このモードは、`--fp16 --fp16_backend apex --fp16_opt_level 01`コマンド ライン引数が渡されると有効になります。

このモードを明示的に構成することもできます。

```json
{
    "amp": {
        "enabled": true,
        "opt_level": "O1"
    }
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

これは[ドキュメント](https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options)です。

<a id='deepspeed-bs'></a>

### Batch Size

バッチサイズを設定するには、次を使用します。


```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}
```

[`Trainer`] は自動的に `train_micro_batch_size_per_gpu` を次の値に設定します。
`args.per_device_train_batch_size`と`train_batch_size`を`args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`に変更します。

値を明示的に設定することもできます。

```json
{
    "train_batch_size": 12,
    "train_micro_batch_size_per_gpu": 4
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

<a id='deepspeed-grad-acc'></a>

### Gradient Accumulation

勾配累積セットを構成するには:

```json
{
    "gradient_accumulation_steps": "auto"
}
```

[`Trainer`] は自動的にそれを `args.gradient_accumulation_steps` の値に設定します。

値を明示的に設定することもできます。

```json
{
    "gradient_accumulation_steps": 3
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

<a id='deepspeed-grad-clip'></a>

### Gradient Clipping

グラデーション グラデーション クリッピング セットを構成するには:

```json
{
    "gradient_clipping": "auto"
}
```

[`Trainer`] は自動的にそれを `args.max_grad_norm` の値に設定します。

値を明示的に設定することもできます。

```json
{
    "gradient_clipping": 1.0
}
```

ただし、[`Trainer`] コマンドライン引数と DeepSpeed を自分で同期することになります。
構成。

<a id='deepspeed-weight-extraction'></a>

### Getting The Model Weights Out

トレーニングを継続し、DeepSpeed の使用を再開する限り、何も心配する必要はありません。 DeepSpeed ストア
fp32 のカスタム チェックポイント オプティマイザー ファイル内のマスターの重み。これは `global_step*/*optim_states.pt` (これは glob
パターン)、通常のチェックポイントの下に保存されます。

**FP16 ウェイト:**

モデルを ZeRO-2 で保存すると、モデルの重みを含む通常の `pytorch_model.bin` ファイルが作成されますが、
これらは重みの fp16 バージョンにすぎません。

ZeRO-3 では、モデルの重みが複数の GPU に分割されるため、状況はさらに複雑になります。
したがって、fp16 を保存するための `Trainer` を取得するには、`"stage3_gather_16bit_weights_on_model_save": true` が必要です。
重みのバージョン。この設定が`False`の場合、`pytorch_model.bin`は作成されません。これは、デフォルトで DeepSpeed の `state_dict` に実際の重みではなくプレースホルダーが含まれるためです。この `state_dict` を保存した場合、ロードし直すことはできません。

```json
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

**FP32 重量:**

fp16 ウェイトはトレーニングを再開するのに適していますが、モデルの微調整が完了し、それを
[モデル ハブ](https://huggingface.co/models) にアクセスするか、fp32 を入手したいと思われる他の人に渡します。
重み。これは大量のメモリを必要とするプロセスであるため、トレーニング中に行うべきではないのが理想的です。
したがって、トレーニングの完了後にオフラインで実行するのが最適です。ただし、必要に応じて、空き CPU が十分にある場合は、
同じトレーニング スクリプトで実行できることを思い出してください。次のセクションでは、両方のアプローチについて説明します。


**ライブ FP32 ウェイト リカバリ:**

モデルが大きく、トレーニングの終了時に空き CPU メモリがほとんど残っていない場合、このアプローチは機能しない可能性があります。

少なくとも 1 つのチェックポイントを保存していて、最新のチェックポイントを使用したい場合は、次の手順を実行できます。

```python
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

`--load_best_model_at_end` class:*~transformers.TrainingArguments* 引数を使用している場合 (最適なモデルを追跡するため)
チェックポイント)、最初に最終モデルを明示的に保存してから、上記と同じことを行うことでトレーニングを終了できます。

```python
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

<Tip>

`load_state_dict_from_zero_checkpoint` が実行されると、`model` はもはや使用できなくなることに注意してください。
同じアプリケーションの DeepSpeed コンテキスト。つまり、deepspeed エンジンを再初期化する必要があります。
`model.load_state_dict(state_dict)` はそこからすべての DeepSpeed マジックを削除します。したがって、これは最後にのみ実行してください
トレーニングの様子。


</Tip>

もちろん、class:*~transformers.Trainer* を使用する必要はなく、上記の例を独自のものに調整することができます。
トレーナー。

何らかの理由でさらに改良したい場合は、重みの fp32 `state_dict` を抽出して適用することもできます。
次の例に示すように、これらは自分で作成します。

```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # already on cpu
model = model.cpu()
model.load_state_dict(state_dict)
```

**オフライン FP32 ウェイト リカバリ:**

DeepSpeed は特別な変換スクリプト`zero_to_fp32.py`を作成し、チェックポイントの最上位に配置します。
フォルダ。このスクリプトを使用すると、いつでも重みを抽出できます。スクリプトはスタンドアロンなので、もう必要ありません。
抽出を行うための設定ファイルまたは `Trainer` が必要です。

チェックポイント フォルダーが次のようになっているとします。

```bash
$ ls -l output_dir/checkpoint-1/
-rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
-rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
-rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
-rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
-rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
-rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
-rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
-rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
-rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
-rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.json
-rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*
```

この例では、DeepSpeed チェックポイント サブフォルダー *global_step1* が 1 つだけあります。したがって、FP32を再構築するには
重みを実行するだけです:

```bash
python zero_to_fp32.py . pytorch_model.bin
```

これだよ。 `pytorch_model.bin`には、複数の GPU から統合された完全な fp32 モデルの重みが含まれるようになります。

スクリプトは、ZeRO-2 または ZeRO-3 チェックポイントを自動的に処理できるようになります。

`python zero_to_fp32.py -h` を実行すると、使用方法の詳細が表示されます。

スクリプトは、ファイル`latest`の内容を使用して deepspeed サブフォルダーを自動検出します。
例には`global_step1`が含まれます。

注: 現在、スクリプトには最終的な fp32 モデルの重みの 2 倍の一般 RAM が必要です。

### ZeRO-3 と Infinity Nuances

ZeRO-3 は、パラメータ シャーディング機能の点で ZeRO-2 とは大きく異なります。

ZeRO-Infinity は ZeRO-3 をさらに拡張し、NVMe メモリやその他の複数の速度とスケーラビリティの向上をサポートします。

モデルに特別な変更を加える必要がなくても正常に動作するようにあらゆる努力が払われてきましたが、特定の点では
状況によっては、次の情報が必要になる場合があります。

#### Constructing Massive Models


DeepSpeed/ZeRO-3 は、既存の RAM に収まらない可能性のある数兆のパラメータを持つモデルを処理できます。そのような場合、
また、初期化をより高速に実行したい場合は、*deepspeed.zero.Init()* を使用してモデルを初期化します。
コンテキスト マネージャー (関数デコレーターでもあります)。次のようになります。

```python
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

ご覧のとおり、これによりランダムに初期化されたモデルが得られます。

事前トレーニングされたモデルを使用したい場合、`model_class.from_pretrained` は次の条件を満たす限りこの機能を有効にします。
`is_deepspeed_zero3_enabled()` は `True` を返します。これは現在、
[`TrainingArguments`] オブジェクト (渡された DeepSpeed 構成ファイルに ZeRO-3 構成が含まれている場合)
セクション。したがって、呼び出しの前に** [`TrainingArguments`] オブジェクトを作成する必要があります。
`from_pretrained`。考えられるシーケンスの例を次に示します。

```python
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

公式のサンプル スクリプトを使用していて、コマンド ライン引数に `--deepspeed ds_config.json` が含まれている場合
ZeRO-3 設定を有効にすると、これがサンプル スクリプトの記述方法であるため、すべてがすでに完了しています。

注: モデルの fp16 重みが単一の GPU のメモリに収まらない場合は、この機能を使用する必要があります。

この方法とその他の関連機能の詳細については、[大規模モデルの構築](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) を参照してください。

また、fp16 で事前訓練されたモデルをロードするときは、`from_pretrained` に使用するように指示する必要があります。
`torch_dtype=torch.float16`。詳細については、[from_pretrained-torch-dtype](#from_pretrained-torch-dtype) を参照してください。

#### Gathering Parameters

複数の GPU 上の ZeRO-3 では、現在の GPU のパラメータでない限り、単一の GPU がすべてのパラメータを持つことはありません。
実行層。したがって、すべてのレイヤーのすべてのパラメーターに一度にアクセスする必要がある場合は、それを行うための特定の方法があります。
ほとんどの場合は必要ありませんが、必要な場合は、[パラメータの収集](https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination) を参照してください。

ただし、いくつかの場所で内部的に使用しています。その例の 1 つは、事前トレーニングされたモデルの重みをロードするときです。
`from_pretrained`。一度に 1 つのレイヤーをロードし、参加しているすべての GPU に即座に分割します。
大規模なモデルでは、メモリの関係で、1 つの GPU にロードしてから複数の GPU に分散することはできません。
制限。

また、ZeRO-3 では、独自のコードを作成し、次のようなモデル パラメーターの重みが発生するとします。

```python
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

`tensor([1.])` にストレスを感じた場合、またはパラメータのサイズが `1` であるというエラーが発生した場合
より大きな多次元形状。これは、パラメーターが分割されており、表示されるのは ZeRO-3 プレースホルダーであることを意味します。

<a id='deepspeed-zero-inference'></a>


### ZeRO Inference

ZeRO Inference は、ZeRO-3 Training と同じ構成を使用します。オプティマイザーとスケジューラーのセクションは必要ありません。で
実際、同じものをトレーニングと共有したい場合は、これらを設定ファイルに残すことができます。彼らはただそうなるだろう
無視されました。

それ以外の場合は、通常の [`TrainingArguments`] 引数を渡すだけです。例えば：

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

唯一重要なことは、ZeRO-2 には何の利点もないため、ZeRO-3 構成を使用する必要があるということです。
ZeRO-3 のみがパラメーターのシャーディングを実行するのに対し、ZeRO-1 は勾配とオプティマイザーの状態をシャーディングするため、推論に役立ちます。

以下は、利用可能なすべての GPU をデプロイする DeepSpeed で`run_translation.py`を実行する例です。


```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --output_dir output_dir \
--do_eval --max_eval_samples 50 --warmup_steps 50  \
--max_source_length 128 --val_max_target_length 128 \
--overwrite_output_dir --per_device_eval_batch_size 4 \
--predict_with_generate --dataset_config "ro-en" --fp16 \
--source_lang en --target_lang ro --dataset_name wmt16 \
--source_prefix "translate English to Romanian: "
```

推論のために、オプティマイザーの状態と勾配によって使用される追加の大きなメモリは必要ないため、
はるかに大きなバッチやシーケンス長を同じハードウェアに適合できる必要があります。

さらに、DeepSpeed は現在、Deepspeed-Inference と呼ばれる関連製品を開発していますが、これとは何の関係もありません。
ZeRO テクノロジーに準拠していますが、代わりにテンソル並列処理を使用して、単一の GPU に収まらないモデルをスケーリングします。これは
現在開発中です。製品が完成したら統合を提供する予定です。


### Memory Requirements

Deepspeed ZeRO はメモリを CPU (および NVMe) にオフロードできるため、フレームワークは、使用されている GPU の数に応じて必要な CPU および GPU メモリの量を知ることができるユーティリティを提供します。

単一の GPU で `bigscience/T0_3B`を微調整するために必要なメモリの量を見積もってみましょう。

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
   15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

したがって、単一の 80 GB GPU で CPU オフロードなしで搭載することも、小さな 8 GB GPU でも最大 60 GB の CPU メモリが必要になることも可能です。 (これはパラメータ、オプティマイザの状態、および勾配のためのメモリであることに注意してください。cuda カーネル、アクティベーション、および一時メモリにはもう少し多くのメモリが必要です。)

次に、コストと速度のトレードオフになります。より小さい GPU を購入またはレンタルした方が安くなります (Deepspeed ZeRO では複数の GPU を使用できるため、GPU の数を減らすこともできます)。しかし、その場合は遅くなります。そのため、何かを実行する速度を気にしなくても、速度の低下は GPU の使用時間に直接影響し、コストが増大するため、どれが最も効果的かを実験して比較してください。

十分な GPU メモリがある場合は、すべてが高速になるため、CPU/NVMe オフロードを必ず無効にしてください。

たとえば、2 つの GPU に対して同じことを繰り返してみましょう。

```bash
$ python -c 'from transformers import AutoModel; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModel.from_pretrained("bigscience/T0_3B"); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 2 GPUs per node.
SW: Model with 2783M total params, 65M largest layer params.
  per CPU  |  per GPU |   Options
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
   70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=1
   62.23GB |   2.84GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    0.74GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=1
   31.11GB |  23.58GB | offload_param=none, offload_optimizer=none, zero_init=0

```

したがって、ここでは、CPU にオフロードせずに 2x 32GB 以上の GPU が必要になります。

詳細については、[メモリ推定ツール](https://deepspeed.readthedocs.io/en/latest/memory.html) を参照してください。


### Filing Issues


ここでは、問題の真相をすぐに解明し、作業のブロックを解除できるよう、問題を報告する方法を説明します。

レポートには必ず次の内容を含めてください。

1. レポート内の完全な Deepspeed 構成ファイル

2. [`Trainer`] を使用している場合はコマンドライン引数、または
   トレーナーのセットアップを自分でスクリプト作成している場合は、[`TrainingArguments`] 引数。しないでください
   [`TrainingArguments`] には無関係なエントリが多数含まれているため、ダンプします。

3. 次の出力:

   ```bash
    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
    ```

4. 可能であれば、問題を再現できる Google Colab ノートブックへのリンクを含めてください。これを使えます
   [ノートブック](https://github.com/stas00/porting/blob/master/transformers/deepspeed/DeepSpeed_on_colab_CLI.ipynb) として
   出発点。

5. 不可能でない限り、カスタムデータセットではなく、常に使用できる標準データセットを使用してください。

6. 可能であれば、既存の [サンプル](https://github.com/huggingface/transformers/tree/main/examples/pytorch) のいずれかを使用して問題を再現してみてください。

- Deepspeed が問題の原因ではないことがよくあります。

  提出された問題の一部は、Deepspeed とは無関係であることが判明しました。それは、Deepspeed がセットアップから削除された後です。
  問題はまだ残っていた。

  したがって、完全に明白でない場合は、DeepSpeed 関連の問題です。
  例外が発生し、DeepSpeed モジュールが関係していることがわかります。まず、DeepSpeed を含まないセットアップを再テストしてください。
  問題が解決しない場合にのみ、Deepspeed について言及し、必要な詳細をすべて提供してください。

- 問題が統合部分ではなく DeepSpeed コアにあることが明らかな場合は、問題を提出してください。
  [Deepspeed](https://github.com/microsoft/DeepSpeed/) を直接使用します。よくわからない場合でも、ご安心ください。
  どちらの問題トラッカーでも問題ありません。投稿されたらそれを判断し、次の場合は別の問題トラッカーにリダイレクトします。
  そうである必要がある。


### Troubleshooting

#### the `deepspeed` process gets killed at startup without a traceback

`deepspeed`プロセスが起動時にトレースバックなしで強制終了された場合、それは通常、プログラムが試行したことを意味します。
システムが持っているよりも多くの CPU メモリを割り当てるか、プロセスが割り当てを許可されているため、OS カーネルがそれを強制終了します。
プロセス。これは、設定ファイルに `offload_optimizer` または `offload_param` が含まれている可能性が高いためです。
どちらも`cpu`にオフロードするように設定されています。 NVMe を使用している場合は、次の環境で実行している場合は NVMe へのオフロードを試してください。
ゼロ-3。 [特定のモデルに必要なメモリ量を見積もる]方法は次のとおりです(https://deepspeed.readthedocs.io/en/latest/memory.html)。

#### training and/or eval/predict loss is `NaN`

これは、bf16 混合精度モードで事前トレーニングされたモデルを取得し、それを fp16 (混合精度の有無にかかわらず) で使用しようとした場合によく発生します。 TPU でトレーニングされたほとんどのモデル、および多くの場合、Google によってリリースされたモデルは、このカテゴリに分類されます (たとえば、ほぼすべての t5 ベースのモデル)。ここでの解決策は、ハードウェアがサポートしている場合 (TPU、Ampere GPU 以降)、fp32 または bf16 を使用することです。

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

ログには、Deepspeed が次のように`OVERFLOW!`を報告していることがわかります。

```
0%|                                                                                                                             | 0/189 [00:00<?, ?it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 262144
  1%|▌                                                                                                                    | 1/189 [00:00<01:26,  2.17it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 262144, reducing to 131072.0
  1%|█▏
 [...]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 14%|████████████████▌                                                                                                   | 27/189 [00:14<01:13,  2.21it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▏                                                                                                  | 28/189 [00:14<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
 15%|█████████████████▊                                                                                                  | 29/189 [00:15<01:13,  2.18it/s]
 [deepscale] OVERFLOW! Rank 0 Skipping step. Attempted loss scale: 1, reducing to 1
[...]
```

これは、Deepspeed 損失スケーラーが損失オーバーフローを克服するスケーリング係数を見つけられないことを意味します。

(ログはここで読みやすくするためにマッサージされています。)

この場合、通常は `initial_scale_power` の値を上げる必要があります。通常、`initial_scale_power: 32` に設定すると問題が解決します。


### Notes

- DeepSpeed には pip でインストール可能な PyPI パッケージがありますが、ハードウェアに最も適合するように、また有効にする必要がある場合は、[ソース](https://github.com/microsoft/deepspeed#installation) からインストールすることを強くお勧めします。
  1 ビット Adam などの特定の機能は、pypi ディストリビューションでは利用できません。
- 🤗 Transformers で DeepSpeed を使用するために [`Trainer`] を使用する必要はありません - 任意のモデルを使用できます
  後者は [DeepSpeed 統合手順](https://www.deepspeed.ai/getting-started/#writing-deepspeed-models) に従って調整する必要があります。

## Non-Trainer Deepspeed Integration

[`~integrations.HfDeepSpeedConfig`] は、Deepspeed を 🤗 Transformers コアに統合するために使用されます
[`Trainer`] を使用しない場合の機能。実行する唯一のことは、Deepspeed ZeRO-3 パラメータ収集を処理し、`from_pretrained`呼び出し中にモデルを複数の GPU に自動的に分割することです。それ以外はすべて自分で行う必要があります。

[`Trainer`] を使用すると、すべてが自動的に処理されます。

[`Trainer`] を使用しない場合、DeepSpeed ZeRO-3 を効率的に導入するには、
モデルをインスタンス化する前に [`~integrations.HfDeepSpeedConfig`] オブジェクトを削除し、そのオブジェクトを生きたままにします。

Deepspeed ZeRO-1 または ZeRO-2 を使用している場合は、`HfDeepSpeedConfig`を使用する必要はまったくありません。

たとえば、事前トレーニングされたモデルの場合は次のようになります。


```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

または、事前トレーニングされていないモデルの場合:


```python
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...}  # deepspeed config object or path to the file
# must run before instantiating the model to detect zero 3
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive
config = AutoConfig.from_pretrained("openai-community/gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

[`Trainer`] 統合を使用していない場合は、完全に独力で行うことになることに注意してください。基本的には、[Deepspeed](https://www.deepspeed.ai/) Web サイトのドキュメントに従ってください。また、設定ファイルを明示的に設定する必要があります。`"auto"`値は使用できず、代わりに実際の値を入力する必要があります。

## HfDeepSpeedConfig

[[autodoc]] integrations.HfDeepSpeedConfig
    - all

### Custom DeepSpeed ZeRO Inference

以下は、単一の GPU にモデルを適合できない場合に、[`Trainer`] を使用せずに DeepSpeed ZeRO 推論を実行する方法の例です。解決策には、追加の GPU の使用、または GPU メモリを CPU メモリにオフロードすることが含まれます。

ここで理解すべき重要なニュアンスは、ZeRO の設計方法により、異なる GPU で異なる入力を並行して処理できるということです。

この例には大量のメモがあり、自己文書化されています。

必ず次のことを行ってください。

1. 十分な GPU メモリがある場合は、CPU オフロードを無効にします (速度が低下するため)。
2. Ampere または新しい GPU を所有している場合は、処理を高速化するために bf16 を有効にします。そのハードウェアがない場合は、bf16 混合精度で事前トレーニングされたモデル (ほとんどの t5 モデルなど) を使用しない限り、fp16 を有効にすることができます。これらは通常、fp16 でオーバーフローし、出力としてガベージが表示されます。


```python
#!/usr/bin/env python

# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py


from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For in-depth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO can process unrelated inputs on each GPU. So for 2 gpus you process 2 inputs at once.
# If you use more GPUs adjust for more.
# And of course if you have just one input to process you then need to pass the same string to both gpus
# If you use only one GPU, then you will have only rank 0.
rank = torch.distributed.get_rank()
if rank == 0:
    text_in = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
elif rank == 1:
    text_in = "Is this review positive or negative? Review: this is the worst restaurant ever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(text_in, return_tensors="pt").to(device=local_rank)
with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True)
text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"rank{rank}:\n   in={text_in}\n  out={text_out}")
```

それを`t0.py`として保存して実行しましょう。

```bash
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

これは非常に基本的な例であり、ニーズに合わせて調整してください。

### `generate` nuances

ZeRO Stage-3 で複数の GPU を使用する場合、`generate(..., synced_gpus=True)`を呼び出して GPU を同期する必要があります。これを行わないと、1 つの GPU が他の GPU より先に生成を終了した場合、残りの GPU が生成を停止した GPU からウェイトのシャードを受信できなくなるため、システム全体がハングします。

`transformers>=4.28` 以降、`synced_gpus` が明示的に指定されていない場合、これらの条件が検出されると自動的に `True` に設定されます。ただし、必要に応じて `synced_gpus` の値をオーバーライドすることもできます。

## Deepspeed 統合のテスト

DeepSpeed 統合を含む PR を送信する場合は、CircleCI PR CI セットアップには GPU がないことに注意してください。そのため、GPU を必要とするテストは別の CI で毎晩のみ実行されます。したがって、PR で緑色の CI レポートが表示されても、DeepSpeed テストが合格したことを意味するわけではありません。

DeepSpeed テストを実行するには、少なくとも以下を実行してください。

```bash
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

モデリングまたは pytorch サンプル コードのいずれかを変更した場合は、Model Zoo テストも実行します。以下はすべての DeepSpeed テストを実行します。

```bash
RUN_SLOW=1 pytest tests/deepspeed
```


## Main DeepSpeed Resources

- [プロジェクトの github](https://github.com/microsoft/deepspeed)
- [使用方法ドキュメント](https://www.deepspeed.ai/getting-started/)
- [API ドキュメント](https://deepspeed.readthedocs.io/en/latest/index.html)
- [ブログ投稿](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

論文:

- [ZeRO: 兆パラメータ モデルのトレーニングに向けたメモリの最適化](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: 10 億規模のモデル トレーニングの民主化](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: 極限スケールの深層学習のための GPU メモリの壁を打ち破る](https://arxiv.org/abs/2104.07857)

最後に、HuggingFace [`Trainer`] は DeepSpeed のみを統合していることを覚えておいてください。
DeepSpeed の使用に関して問題や質問がある場合は、[DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed/issues) に問題を提出してください。
