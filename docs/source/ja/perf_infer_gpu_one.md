<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Inference on a Single GPU

このガイドに加えて、[1つのGPUでのトレーニングガイド](perf_train_gpu_one)と[CPUでの推論ガイド](perf_infer_cpu)に関連する情報があります。

## Flash Attention 2

<Tip>

この機能は実験的であり、将来のバージョンで大幅に変更される可能性があります。たとえば、Flash Attention 2 APIは近い将来`BetterTransformer` APIに移行するかもしれません。

</Tip>

Flash Attention 2は、トランスフォーマーベースのモデルのトレーニングと推論速度を大幅に高速化できます。Flash Attention 2は、Tri Dao氏によって[公式のFlash Attentionリポジトリ](https://github.com/Dao-AILab/flash-attention)で導入されました。Flash Attentionに関する科学論文は[こちら](https://arxiv.org/abs/2205.14135)で見ることができます。

Flash Attention 2を正しくインストールするには、上記のリポジトリに記載されているインストールガイドに従ってください。

以下のモデルに対してFlash Attention 2をネイティブサポートしています：

- Llama
- Falcon

さらに多くのモデルにFlash Attention 2のサポートを追加することをGitHubで提案することもでき、変更を統合するためにプルリクエストを開くこともできます。サポートされているモデルは、パディングトークンを使用してトレーニングを含む、推論とトレーニングに使用できます（現在の`BetterTransformer` APIではサポートされていない）。

<Tip>

Flash Attention 2は、モデルのdtypeが`fp16`または`bf16`の場合にのみ使用でき、NVIDIA-GPUデバイスでのみ実行されます。この機能を使用する前に、モデルを適切なdtypeにキャストし、サポートされているデバイスにロードしてください。

</Tip>

### Quick usage

モデルでFlash Attention 2を有効にするには、`from_pretrained`の引数に`attn_implementation="flash_attention_2"`を追加します。


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

こちらは、生成または微調整のために使用するテキストです。

### Expected speedups

特に長いシーケンスに対して、微調整と推論の際には、かなりの高速化が期待できます。ただし、Flash Attentionはパディングトークンを使用してアテンションスコアを計算しないため、シーケンスにパディングトークンが含まれる場合、バッチ推論においてアテンションスコアを手動でパッド/アンパッドする必要があり、パディングトークンを含むバッチ生成の大幅な遅延が発生します。

これを克服するために、トレーニング中にシーケンスにパディングトークンを使用せずにFlash Attentionを使用する必要があります（たとえば、データセットをパックすることにより、シーケンスを最大シーケンス長に達するまで連結することなど）。ここに[例](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py#L516)が提供されています。

以下は、パディングトークンのない場合に、シーケンス長が4096の[tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b)に対する単純なフォワードパスの予想される高速化です。さまざまなバッチサイズが示されています：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png">
</div>

以下は、パディングトークンのない場合に、シーケンス長が4096の[`meta-llama/Llama-7b-hf`](https://hf.co/meta-llama/Llama-7b-hf)に対する単純なフォワードパスの予想される高速化です。さまざまなバッチサイズが示されています：

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-7b-inference-large-seqlen.png">
</div>

パディングトークンを含むシーケンス（パディングトークンを使用してトレーニングまたは生成する）の場合、アテンションスコアを正しく計算するために入力シーケンスをアンパッド/パッドする必要があります。比較的小さいシーケンス長の場合、純粋なフォワードパスではパディングトークンが30%未満しか埋められていないため、これはわずかな高速化をもたらします。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png">
</div>

しかし、大きなシーケンス長の場合、純粋な推論（トレーニングも含む）には興味深い高速化が得られます。

Flash Attentionは、アテンション計算をよりメモリ効率の良いものにし、大きなシーケンス長でのCUDA OOMの問題を回避できるようにします。大きなシーケンス長に対して最大20のメモリ削減をもたらすことがあります。詳細については、[公式のFlash Attentionリポジトリ](https://github.com/Dao-AILab/flash-attention)をご覧ください。

<div style="text-align: center">
<img src="https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-large-seqlen-padding.png">
</div>


### Advanced usage

この機能をモデルの最適化に多くの既存の機能と組み合わせることができます。以下にいくつかの例を示します：

### Combining Flash Attention 2 and 8-bit models

この機能を8ビットの量子化と組み合わせることができます：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
)
```

### Combining Flash Attention 2 and 4-bit models

この機能を 4 ビットの量子化と組み合わせることができます：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)
```

### Combining Flash Attention 2 and PEFT

この機能を使用して、Flash Attention 2をベースにアダプターをトレーニングする際にPEFTを組み合わせることができます。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)

lora_config = LoraConfig(
    r=8,
    task_type="CAUSAL_LM"
)

model.add_adapter(lora_config)

... # train your model
```

## BetterTransformer

[BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview)は、🤗 TransformersモデルをPyTorchネイティブの高速パス実行に変換します。これにより、Flash Attentionなどの最適化されたカーネルが内部で呼び出されます。

BetterTransformerは、テキスト、画像、およびオーディオモデルの単一およびマルチGPUでの高速な推論をサポートしています。

<Tip>

Flash Attentionは、fp16またはbf16のdtypeを使用するモデルにのみ使用できます。BetterTransformerを使用する前に、モデルを適切なdtypeにキャストしてください。

</Tip>

### Encoder models

PyTorchネイティブの[`nn.MultiHeadAttention`](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)アテンション高速パス、BetterTransformerと呼ばれるものは、[🤗 Optimumライブラリ](https://huggingface.co/docs/optimum/bettertransformer/overview)の統合を通じてTransformersと一緒に使用できます。

PyTorchのアテンション高速パスを使用すると、カーネルフュージョンと[ネストされたテンソル](https://pytorch.org/docs/stable/nested.html)の使用により、推論を高速化できます。詳細なベンチマーク情報は[このブログ記事](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2)にあります。

[`optimum`](https://github.com/huggingface/optimum)パッケージをインストールした後、推論中にBetter Transformerを使用するには、関連する内部モジュールを呼び出すことで置き換える必要があります[`~PreTrainedModel.to_bettertransformer`]:


```python
model = model.to_bettertransformer()
```

メソッド [`~PreTrainedModel.reverse_bettertransformer`] は、モデルを保存する前に使用すべきで、標準のトランスフォーマーモデリングを使用するためのものです：

```python
model = model.reverse_bettertransformer()
model.save_pretrained("saved_model")
```

BetterTransformer APIを使ったエンコーダーモデルの可能性について詳しく知るには、[このブログポスト](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2)をご覧ください。

### Decoder models

テキストモデル、特にデコーダーベースのモデル（GPT、T5、Llamaなど）にとって、BetterTransformer APIはすべての注意操作を[`torch.nn.functional.scaled_dot_product_attention`オペレーター](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention)（SDPA）を使用するように変換します。このオペレーターはPyTorch 2.0以降でのみ利用可能です。

モデルをBetterTransformerに変換するには、以下の手順を実行してください：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
# convert the model to BetterTransformer
model.to_bettertransformer()

# Use it for training or inference
```

SDPAは、ハードウェアや問題のサイズに応じて[Flash Attention](https://arxiv.org/abs/2205.14135)カーネルを使用することもできます。Flash Attentionを有効にするか、特定の設定（ハードウェア、問題サイズ）で使用可能かどうかを確認するには、[`torch.nn.attention.sdpa_kernel`](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)をコンテキストマネージャとして使用します。


```diff
import torch
+ from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

もしトレースバックにバグが表示された場合

```bash
RuntimeError: No available kernel.  Aborting execution.
```

Flash Attention の広範なカバレッジを持つかもしれない PyTorch のナイトリーバージョンを試してみることをお勧めします。

```bash
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Or make sure your model is correctly casted in float16 or bfloat16

モデルが正しくfloat16またはbfloat16にキャストされていることを確認してください。

Have a look at [this detailed blogpost](https://pytorch.org/blog/out-of-the-box-acceleration/) to read more about what is possible to do with `BetterTransformer` + SDPA API.

`BetterTransformer` + SDPA APIを使用して何が可能かについて詳しく読むには、[この詳細なブログポスト](https://pytorch.org/blog/out-of-the-box-acceleration/)をご覧ください。

## `bitsandbytes` integration for FP4 mixed-precision inference

FP4混合精度推論のための`bitsandbytes`統合

You can install `bitsandbytes` and benefit from easy model compression on GPUs. Using FP4 quantization you can expect to reduce up to 8x the model size compared to its native full precision version. Check out below how to get started.

`bitsandbytes`をインストールし、GPUで簡単なモデルの圧縮を利用できます。FP4量子化を使用すると、ネイティブのフルプレシジョンバージョンと比較してモデルサイズを最大8倍削減できることが期待できます。以下を確認して、どのように始めるかをご覧ください。

<Tip>

Note that this feature can also be used in a multi GPU setup.

この機能は、マルチGPUセットアップでも使用できることに注意してください。

</Tip>

### Requirements [[requirements-for-fp4-mixedprecision-inference]]

- Latest `bitsandbytes` library
`pip install bitsandbytes>=0.39.0`

- Install latest `accelerate` from source
`pip install git+https://github.com/huggingface/accelerate.git`

- Install latest `transformers` from source
`pip install git+https://github.com/huggingface/transformers.git`


### Running FP4 models - single GPU setup - Quickstart

以下のコードを実行することで、簡単に単一のGPUでFP4モデルを実行できます:


```py
from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

注意: `device_map`はオプションですが、推論時に `device_map = 'auto'` を設定することが推奨されています。これにより、利用可能なリソースに効率的にモデルがディスパッチされます。

### Running FP4 models - multi GPU setup

混合4ビットモデルを複数のGPUにロードする方法は、単一GPUセットアップと同じです（単一GPUセットアップと同じコマンドです）：

```py
model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
```

しかし、`accelerate`を使用して、各GPUに割り当てるGPU RAMを制御することができます。以下のように、`max_memory`引数を使用します：


```py
max_memory_mapping = {0: "600MB", 1: "1GB"}
model_name = "bigscience/bloom-3b"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_4bit=True, max_memory=max_memory_mapping
)
```

この例では、最初のGPUは600MBのメモリを使用し、2番目のGPUは1GBを使用します。

### Advanced usage

このメソッドのさらなる高度な使用法については、[量子化](main_classes/quantization)のドキュメンテーションページをご覧ください。

## `bitsandbytes` integration for Int8 mixed-precision matrix decomposition

<Tip>

この機能は、マルチGPU環境でも使用できます。

</Tip>

論文[`LLM.int8()：スケーラブルなTransformer向けの8ビット行列乗算`](https://arxiv.org/abs/2208.07339)によれば、Hugging Face統合がHub内のすべてのモデルでわずか数行のコードでサポートされています。このメソッドは、半精度（`float16`および`bfloat16`）の重みの場合に`nn.Linear`サイズを2倍、単精度（`float32`）の重みの場合は4倍に縮小し、外れ値に対してほとんど影響を与えません。

![HFxbitsandbytes.png](https://cdn-uploads.huggingface.co/production/uploads/1659861207959-62441d1d9fdefb55a0b7d12c.png)

Int8混合精度行列分解は、行列乗算を2つのストリームに分割することによって動作します：(1) システマティックな特徴外れ値ストリームがfp16で行列乗算（0.01%）、(2) int8行列乗算の通常のストリーム（99.9%）。この方法を使用すると、非常に大きなモデルに対して予測の劣化なしにint8推論が可能です。
このメソッドの詳細については、[論文](https://arxiv.org/abs/2208.07339)または[この統合に関するブログ記事](https://huggingface.co/blog/hf-bitsandbytes-integration)をご確認ください。

![MixedInt8.gif](https://cdn-uploads.huggingface.co/production/uploads/1660567469965-62441d1d9fdefb55a0b7d12c.gif)

なお、この機能を使用するにはGPUが必要であり、カーネルはGPU専用にコンパイルされている必要があります。この機能を使用する前に、モデルの1/4（またはハーフ精度の重みの場合は1/2）を保存するのに十分なGPUメモリがあることを確認してください。
このモジュールを使用する際のヘルプに関する詳細は、以下のノートをご覧いただくか、[Google Colabのデモ](#colab-demos)をご覧ください。

### Requirements [[requirements-for-int8-mixedprecision-matrix-decomposition]]

- `bitsandbytes<0.37.0`を使用する場合、NVIDIA GPUを使用していることを確認し、8ビットテンソルコアをサポートしていることを確認してください（Turing、Ampere、またはそれ以降のアーキテクチャー、例：T4、RTX20s RTX30s、A40-A100など）。`bitsandbytes>=0.37.0`の場合、すべてのGPUがサポートされるはずです。
- 正しいバージョンの`bitsandbytes`をインストールするには、次のコマンドを実行してください：
`pip install bitsandbytes>=0.31.5`
- `accelerate`をインストールします：
`pip install accelerate>=0.12.0`


### Running mixed-Int8 models - single GPU setup

必要なライブラリをインストールした後、ミックス 8 ビットモデルを読み込む方法は次の通りです：

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

以下はシンプルな例です：

* `pipeline()` 関数の代わりに、モデルの `generate()` メソッドを使用することをお勧めします。`pipeline()` 関数を使用して推論することは可能ですが、混合8ビットモデルに最適化されておらず、`generate()` メソッドを使用するよりも遅くなります。また、一部のサンプリング戦略（例：ヌクレウスサンプリング）は、`pipeline()` 関数では混合8ビットモデルではサポートされていません。
* すべての入力をモデルと同じデバイスに配置してください。


```py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "bigscience/bloom-2b5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

### Running mixed-int8 models - multi GPU setup

複数のGPUに混合8ビットモデルをロードする方法は、次の通りです（シングルGPUセットアップと同じコマンドです）：

```py
model_name = "bigscience/bloom-2b5"
model_8bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
```

`accelerate`を使用して各GPUに割り当てるGPU RAMを制御する際には、以下のように`max_memory`引数を使用します：


```py
max_memory_mapping = {0: "1GB", 1: "2GB"}
model_name = "bigscience/bloom-3b"
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory_mapping
)
```

In this example, the first GPU will use 1GB of memory and the second 2GB.

### Colab demos

この方法を使用すると、以前のGoogle Colabでは推論できなかったモデルに対して推論を行うことができます。以下は、Google Colabで8ビット量子化を使用してT5-11b（fp32で42GB）を実行するデモのリンクです：

[![Open In Colab: T5-11b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

また、BLOOM-3Bのデモもご覧いただけます：

[![Open In Colab: BLOOM-3b demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)

## Advanced usage: mixing FP4 (or Int8) and BetterTransformer

異なる方法を組み合わせて、モデルの最適なパフォーマンスを得ることができます。例えば、BetterTransformerを使用してFP4ミックスプレシジョン推論とフラッシュアテンションを組み合わせることができます。


```py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```