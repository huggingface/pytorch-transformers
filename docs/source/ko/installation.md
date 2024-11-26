<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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

# 설치방법[[installation]]

🤗 Transformers를 사용 중인 딥러닝 라이브러리에 맞춰 설치하고, 캐시를 구성하거나 선택적으로 오프라인에서도 실행할 수 있도록 🤗 Transformers를 설정하는 방법을 배우겠습니다.

🤗 Transformers는 Python 3.6+, PyTorch 1.1.0+, TensorFlow 2.0+ 및 Flax에서 테스트되었습니다. 딥러닝 라이브러리를 설치하려면 아래 링크된 저마다의 공식 사이트를 참고해주세요.

* [PyTorch](https://pytorch.org/get-started/locally/) 설치하기
* [TensorFlow 2.0](https://www.tensorflow.org/install/pip) 설치하기
* [Flax](https://flax.readthedocs.io/en/latest/) 설치하기

## pip으로 설치하기[[install-with-pip]]

🤗 Transformers를 [가상 환경](https://docs.python.org/3/library/venv.html)에 설치하는 것을 추천드립니다. Python 가상 환경에 익숙하지 않다면, 이 [가이드](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)를 참고하세요. 가상 환경을 사용하면 서로 다른 프로젝트들을 보다 쉽게 관리할 수 있고, 의존성 간의 호환성 문제를 방지할 수 있습니다.

먼저 프로젝트 디렉토리에서 가상 환경을 만들어 줍니다.

```bash
python -m venv .env
```

가상 환경을 활성화해주세요. Linux나 MacOS의 경우:

```bash
source .env/bin/activate
```
Windows의 경우:

```bash
.env/Scripts/activate
```

이제 🤗 Transformers를 설치할 준비가 되었습니다. 다음 명령을 입력해주세요.

```bash
pip install transformers
```

CPU만 써도 된다면, 🤗 Transformers와 딥러닝 라이브러리를 단 1줄로 설치할 수 있습니다. 예를 들어 🤗 Transformers와 PyTorch의 경우:

```bash
pip install transformers[torch]
```

🤗 Transformers와 TensorFlow 2.0의 경우:

```bash
pip install transformers[tf-cpu]
```

🤗 Transformers와 Flax의 경우:

```bash
pip install transformers[flax]
```

마지막으로 🤗 Transformers가 제대로 설치되었는지 확인할 차례입니다. 사전훈련된 모델을 다운로드하는 코드입니다.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

라벨과 점수가 출력되면 잘 설치된 것입니다.

```bash
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

## 소스에서 설치하기[[install-from-source]]

🤗 Transformers를 소스에서 설치하려면 아래 명령을 실행하세요.

```bash
pip install git+https://github.com/huggingface/transformers
```

위 명령은 최신이지만 (안정적인) `stable` 버전이 아닌 실험성이 짙은 `main` 버전을 설치합니다. `main` 버전은 개발 현황과 발맞추는데 유용합니다. 예시로 마지막 공식 릴리스 이후 발견된 버그가 패치되었지만, 새 릴리스로 아직 롤아웃되지는 않은 경우를 들 수 있습니다. 바꿔 말하면 `main` 버전이 안정성과는 거리가 있다는 뜻이기도 합니다. 저희는 `main` 버전을 사용하는데 문제가 없도록 노력하고 있으며, 대부분의 문제는 대개 몇 시간이나 하루 안에 해결됩니다. 만약 문제가 발생하면 [이슈](https://github.com/huggingface/transformers/issues)를 열어주시면 더 빨리 해결할 수 있습니다!

전과 마찬가지로 🤗 Transformers가 제대로 설치되었는지 확인할 차례입니다.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

## 수정 가능한 설치[[editable-install]]

수정 가능한 설치가 필요한 경우는 다음과 같습니다.

* `main` 버전의 소스 코드를 사용하기 위해
* 🤗 Transformers에 기여하고 싶어서 코드의 변경 사항을 테스트하기 위해

리포지터리를 복제하고 🤗 Transformers를 설치하려면 다음 명령을 입력해주세요.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```

위 명령은 리포지터리를 복제한 위치의 폴더와 Python 라이브러리의 경로를 연결시킵니다. Python이 일반 라이브러리 경로 외에 복제한 폴더 내부를 확인할 것입니다. 예를 들어 Python 패키지가 일반적으로 `~/anaconda3/envs/main/lib/python3.7/site-packages/`에 설치되어 있는데, 명령을 받은 Python이 이제 복제한 폴더인 `~/transformers/`도 검색하게 됩니다.

<Tip warning={true}>

라이브러리를 계속 사용하려면 `transformers` 폴더를 꼭 유지해야 합니다.

</Tip>

복제본은 최신 버전의 🤗 Transformers로 쉽게 업데이트할 수 있습니다.

```bash
cd ~/transformers/
git pull
```

Python 환경을 다시 실행하면 업데이트된 🤗 Transformers의 `main` 버전을 찾아낼 것입니다.

## conda로 설치하기[[install-with-conda]]

`conda-forge` conda 채널에서 설치할 수 있습니다.

```bash
conda install conda-forge::transformers
```

## 캐시 구성하기[[cache-setup]]

사전훈련된 모델은 다운로드된 후 로컬 경로 `~/.cache/huggingface/hub`에 캐시됩니다. 셸 환경 변수 `TRANSFORMERS_CACHE`의 기본 디렉터리입니다. Windows의 경우 기본 디렉터리는 `C:\Users\username\.cache\huggingface\hub`입니다. 아래의 셸 환경 변수를 (우선 순위) 순서대로 변경하여 다른 캐시 디렉토리를 지정할 수 있습니다.

1. 셸 환경 변수 (기본): `HF_HUB_CACHE` 또는 `TRANSFORMERS_CACHE`
2. 셸 환경 변수: `HF_HOME`
3. 셸 환경 변수: `XDG_CACHE_HOME` + `/huggingface`

<Tip>

과거 🤗 Transformers에서 쓰였던 셸 환경 변수 `PYTORCH_TRANSFORMERS_CACHE` 또는 `PYTORCH_PRETRAINED_BERT_CACHE`이 설정되있다면, 셸 환경 변수 `TRANSFORMERS_CACHE`을 지정하지 않는 한 우선 사용됩니다.

</Tip>

## 오프라인 모드[[offline-mode]]

🤗 Transformers를 로컬 파일만 사용하도록 해서 방화벽 또는 오프라인 환경에서 실행할 수 있습니다. 활성화하려면 `HF_HUB_OFFLINE=1` 환경 변수를 설정하세요.

<Tip>

`HF_DATASETS_OFFLINE=1` 환경 변수를 설정하여 오프라인 훈련 과정에 [🤗 Datasets](https://huggingface.co/docs/datasets/)을 추가할 수 있습니다.

</Tip>

예를 들어 외부 기기 사이에 방화벽을 둔 일반 네트워크에서 평소처럼 프로그램을 다음과 같이 실행할 수 있습니다.

```bash
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

오프라인 기기에서 동일한 프로그램을 다음과 같이 실행할 수 있습니다.

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python examples/pytorch/translation/run_translation.py --model_name_or_path google-t5/t5-small --dataset_name wmt16 --dataset_config ro-en ...
```

이제 스크립트는 로컬 파일에 한해서만 검색할 것이므로, 스크립트가 중단되거나 시간이 초과될 때까지 멈춰있지 않고 잘 실행될 것입니다.

### 오프라인용 모델 및 토크나이저 만들어두기[[fetch-models-and-tokenizers-to-use-offline]]

Another option for using 🤗 Transformers offline is to download the files ahead of time, and then point to their local path when you need to use them offline. There are three ways to do this:
🤗 Transformers를 오프라인으로 사용하는 또 다른 방법은 파일을 미리 다운로드한 다음, 오프라인일 때 사용할 로컬 경로를 지정해두는 것입니다. 3가지 중 편한 방법을 고르세요.

* [Model Hub](https://huggingface.co/models)의 UI를 통해 파일을 다운로드하려면 ↓ 아이콘을 클릭하세요.

    ![download-icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/download-icon.png)

* [`PreTrainedModel.from_pretrained`]와 [`PreTrainedModel.save_pretrained`] 워크플로를 활용하세요.

    1. 미리 [`PreTrainedModel.from_pretrained`]로 파일을 다운로드해두세요.

    ```py
    >>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")
    ```

    2. [`PreTrainedModel.save_pretrained`]로 지정된 경로에 파일을 저장해두세요.

    ```py
    >>> tokenizer.save_pretrained("./your/path/bigscience_t0")
    >>> model.save_pretrained("./your/path/bigscience_t0")
    ```

    3. 이제 오프라인일 때 [`PreTrainedModel.from_pretrained`]로 저장해뒀던 파일을 지정된 경로에서 다시 불러오세요.

    ```py
    >>> tokenizer = AutoTokenizer.from_pretrained("./your/path/bigscience_t0")
    >>> model = AutoModel.from_pretrained("./your/path/bigscience_t0")
    ```

* [huggingface_hub](https://github.com/huggingface/huggingface_hub/tree/main/src/huggingface_hub) 라이브러리를 활용해서 파일을 다운로드하세요.

    1. 가상환경에 `huggingface_hub` 라이브러리를 설치하세요.

    ```bash
    python -m pip install huggingface_hub
    ```

    2. [`hf_hub_download`](https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub) 함수로 파일을 특정 위치에 다운로드할 수 있습니다. 예를 들어 아래 명령은 [T0](https://huggingface.co/bigscience/T0_3B) 모델의 `config.json` 파일을 지정된 경로에 다운로드합니다.

    ```py
    >>> from huggingface_hub import hf_hub_download

    >>> hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./your/path/bigscience_t0")
    ```

파일을 다운로드하고 로컬에 캐시 해놓고 나면, 나중에 불러와 사용할 수 있도록 로컬 경로를 지정해두세요.

```py
>>> from transformers import AutoConfig

>>> config = AutoConfig.from_pretrained("./your/path/bigscience_t0/config.json")
```

<Tip>

Hub에 저장된 파일을 다운로드하는 방법을 더 자세히 알아보려면 [Hub에서 파일 다운로드하기](https://huggingface.co/docs/hub/how-to-downstream) 섹션을 참고해주세요.

</Tip>
