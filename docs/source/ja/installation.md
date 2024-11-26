<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# インストール

使用しているDeep Learningライブラリに対して、🤗 Transformersをインストールしてキャッシュを設定、そしてオプションでオフラインで実行できるように 🤗 Transformersを設定します。

🤗 TransformersはPython 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+, Flaxで動作確認しています。 使用しているDeep Learningライブラリに合わせて、以下のインストール方法に従ってください:

* [PyTorch](https://pytorch.org/get-started/locally/)のインストール手順。
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip)のインストール手順。
* [Flax](https://flax.readthedocs.io/en/latest/)のインストール手順。

## pipでのインストール

🤗 Transformersを[仮想環境](https://docs.python.org/3/library/venv.html)にインストールする必要があります。 もし、Pythonの仮想環境に馴染みがない場合は、この[ガイド](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)をご覧ください。仮想環境によって異なるプロジェクトの管理がより簡単になり、依存関係間の互換性の問題を回避できます。

まず、プロジェクトディレクトリに仮想環境を作成することから始めましょう:

```bash
python -m venv .env
```

仮想環境を起動しましょう。LinuxとMacOsの場合は以下のコマンドで起動します:

```bash
source .env/bin/activate
```
Windowsで仮想環境を起動します

```bash
.env/Scripts/activate
```

これで、次のコマンドで🤗 Transformersをインストールする準備が整いました:

```bash
pip install transformers
```

CPU対応のみ必要な場合、🤗 TransformersとDeep Learningライブラリを1行でインストールできるようになっていて便利です。例えば、🤗 TransformersとPyTorchを以下のように一緒にインストールできます:

```bash
pip install transformers[torch]
```

🤗 TransformersとTensorFlow 2.0:

```bash
pip install transformers[tf-cpu]
```

🤗 TransformersとFlax:

```bash
pip install transformers[flax]
```

最後に、以下のコマンドを実行することで🤗 Transformersが正しくインストールされているかを確認します。学習済みモデルがダウンロードされます:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

その後、ラベルとスコアが出力されます:

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## ソースからのインストール

以下のコマンドでソースから🤗 Transformersをインストールします:

```bash
pip install git+https://github.com/huggingface/transformers
```

このコマンドは最新の安定版ではなく、開発における最新の`main`バージョンをインストールします。`main`バージョンは最新の開発状況に対応するのに便利です。例えば、最後の公式リリース以降にバグが修正されたが、新しいリリースがまだ展開されていない場合などです。しかし、これは`main`バージョンが常に安定しているとは限らないことを意味します。私たちは`main`バージョンの運用を維持するよう努め、ほとんどの問題は通常、数時間から1日以内に解決されます。もし問題に遭遇した場合は、より早く修正できるように[Issue](https://github.com/huggingface/transformers/issues)を作成してください！

以下のコマンドを実行して、🤗 Transformersが正しくインストールされているかどうかを確認します:

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## 編集可能なインストール

必要に応じて、編集可能なインストールをします:

* ソースコードの`main`バージョンを使います。
* 🤗 Transformersにコントリビュートし、コードの変更をテストする必要があります。

以下のコマンドでレポジトリをクローンして、🤗 Transformersをインストールします:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

上記のコマンドは、レポジトリをクローンしたフォルダとPythonのライブラリをパスをリンクします。Pythonは通常のライブラリパスに加えて、あなたがクローンしたフォルダの中も見るようになります。例えば、Pythonパッケージが通常、`~/anaconda3/envs/main/lib/python3.7/site-packages/`にインストールされている場合、Pythonはクローンしたフォルダも検索するようになります: `~/transformers/`.

<Tip warning={true}>

ライブラリーを使い続けたい場合は、transformersフォルダーを保持しつづける必要があります。

</Tip>

これで、次のコマンドで簡単にクローンを🤗 Transformersの最新版に更新できます:

```bash
cd ~/transformers/
git pull
```

Python環境は次回の実行時に🤗 Transformersの`main`バージョンを見つけるようになります。

## condaでのインストール

`conda-forge`のcondaチャンネルからインストールします:

```bash
conda install conda-forge::transformers
```

## キャッシュの設定

学習済みモデルはダウンロードされ、ローカルにキャッシュされます: `~/.cache/huggingface/hub`. これはシェル環境変数`TRANSFORMERS_CACHE`で指定されるデフォルトのディレクトリです。Windowsでは、デフォルトのディレクトリは`C:\Users\username\.cache\huggingface\hub`になっています。異なるキャッシュディレクトリを指定するために、以下のシェル環境変数を変更することが可能です。優先度は以下の順番に対応します:

1. シェル環境変数 (デフォルト): `HF_HUB_CACHE` または `TRANSFORMERS_CACHE`.
2. シェル環境変数: `HF_HOME`.
3. シェル環境変数: `XDG_CACHE_HOME` + `/huggingface`.

<Tip>

もし、以前のバージョンのライブラリを使用していた人で、`PYTORCH_TRANSFORMERS_CACHE`または`PYTORCH_PRETRAINED_BERT_CACHE`を設定していた場合、シェル環境変数`TRANSFORMERS_CACHE`を指定しない限り🤗 Transformersはこれらのシェル環境変数を使用します。

</Tip>

## オフラインモード

🤗 Transformersはローカルファイルのみを使用することでファイアウォールやオフラインの環境でも動作させることができます。この動作を有効にするためには、環境変数`HF_HUB_OFFLINE=1`を設定します。

<Tip>

環境変数`HF_DATASETS_OFFLINE=1`を設定し、オフライントレーニングワークフローに[🤗 Datasets](https://huggingface.co/docs/datasets/)を追加します。

</Tip>

例えば、外部インスタンスに対してファイアウォールで保護された通常のネットワーク上でプログラムを実行する場合、通常以下のようなコマンドで実行することになります:

```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

オフラインインスタンスでこの同じプログラムを実行します:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

このスクリプトは、ローカルファイルのみを検索することが分かっているので、ハングアップしたりタイムアウトを待ったりすることなく実行されるはずです。

### オフラインで使用するためにモデルやトークナイザーを取得する

オフラインで🤗 Transformersを使用するもう1つの方法は、前もってファイルをダウンロードしておき、オフラインで使用する必要があるときにそのローカルパスを指定することです。これには3つの方法があります:

* [Model Hub](https://huggingface.co/models)のユーザーインターフェース上から↓アイコンをクリックしてファイルをダウンロードする方法。

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* [`PreTrainedModel.from_pretrained`]および[`PreTrainedModel.save_pretrained`]のワークフローを使用する方法:

    1. [`PreTrainedModel.from_pretrained`]で前もってファイルをダウンロードします:

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. [`PreTrainedModel.save_pretrained`]で指定されたディレクトリにファイルを保存しておきます:

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. オフラインにある時、[`PreTrainedModel.from_pretrained`]に指定したディレクトリからファイルをリロードします:

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* プログラム的に[huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub)ライブラリを用いて、ファイルをダウンロードする方法:

    1. 仮想環境に`huggingface_hub`ライブラリをインストールします:

    ```bash
    python -m pip install huggingface_hub
    ```

    2. 指定のパスにファイルをダウンロードするために、[`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub)関数を使用します。例えば、以下のコマンドで、[T0](https://huggingface.co/bigscience/T0_3B)モデルの`config.json`ファイルを指定のパスにダウンロードできます:

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

ファイルがダウンロードされ、ローカルにキャッシュされたら、そのローカルパスを指定してファイルをロードして使用します:

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Hubに保存されているファイルをダウンロードする方法の詳細については、[How to download files from the Hub](https://huggingface.co/docs/hub/how-to-downstream)セクションを参照してください。

</Tip>
