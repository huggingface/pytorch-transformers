<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSpeed[[deepspeed]]

[DeepSpeed](https://www.deepspeed.ai/)는 분산 학습 메모리를 효율적이고 빠르게 만드는 PyTorch 최적화 라이브러리입니다. 그 핵심은 대규모 모델을 규모에 맞게 훈련할 수 있는 [Zero Redundancy Optimizer(ZeRO)](https://hf.co/papers/1910.02054)입니다. ZeRO는 여러 단계로 작동합니다:

* ZeRO-1, GPU 간 최적화 상태 분할
* ZeRO-2, GPU 간 그레이디언트 분할
* ZeRO-3, GPU 간 매개변수 분할

GPU가 제한된 환경에서 ZeRO는 최적화 메모리와 계산을 GPU에서 CPU로 오프로드하여 단일 GPU에 대규모 모델을 장착하고 훈련할 수 있습니다. DeepSpeed는 모든 ZeRO 단계 및 오프로딩을 위해 Transformers [`Trainer`] 클래스와 통합되어 있습니다. 구성 파일을 제공하거나 제공된 템플릿을 사용하기만 하면 됩니다. 추론의 경우, Transformers는 대용량 모델을 가져올 수 있으므로 ZeRO-3 및 오프로딩을 지원합니다.

이 가이드에서는 DeepSpeed 트레이닝을 배포하는 방법, 활성화할 수 있는 기능, 다양한 ZeRO 단계에 대한 구성 파일 설정 방법, 오프로딩, 추론 및 [`Trainer`] 없이 DeepSpeed를 사용하는 방법을 안내해 드립니다.

## 설치[[installation]]

DeepSpeed는 PyPI 또는 Transformers에서 설치할 수 있습니다(자세한 설치 옵션은 DeepSpeed [설치 상세사항](https://www.deepspeed.ai/tutorials/advanced-install/) 또는 GitHub [README](https://github.com/microsoft/deepspeed#installation)를 참조하세요).

<Tip>

DeepSpeed를 설치하는 데 문제가 있는 경우 [DeepSpeed CUDA 설치](../debugging#deepspeed-cuda-installation) 가이드를 확인하세요. DeepSpeed에는 pip 설치 가능한 PyPI 패키지로 설치할 수 있지만, 하드웨어에 가장 잘 맞고 PyPI 배포판에서는 제공되지 않는 1비트 Adam과 같은 특정 기능을 지원하려면 [소스에서 설치하기](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source)를 적극 권장합니다.

</Tip>

<hfoptions id="install">
<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>
<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>
</hfoptions>

## 메모리 요구량[[memory-requirements]]

시작하기 전에 모델에 맞는 충분한 GPU 및 CPU 메모리가 있는지 확인하는 것이 좋습니다. DeepSpeed는 필요한 CPU/GPU 메모리를 추정할 수 있는 도구를 제공합니다. 예를 들어, 단일 GPU에서 [bigscience/T0_3B](bigscience/T0_3B) 모델의 메모리 요구 사항을 추정할 수 있습니다:

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

즉, CPU 오프로드가 없는 단일 80GB GPU 또는 오프로드 할 8GB GPU와 최대 60GB CPU가 필요합니다 (이는 매개변수, 최적화 상태 및 그레이디언트에 대한 메모리 요구 사항일 뿐이며 CUDA 커널 및 활성화에는 조금 더 필요합니다). 또한 더 작은 GPU를 대여하거나 구입하는 것이 더 저렴하지만 모델을 훈련하는 데 시간이 더 오래 걸리므로 비용과 속도 간의 균형을 고려해야 합니다.

GPU 메모리가 충분하다면 CPU/NVMe 오프로드를 비활성화하여 모든 작업을 더 빠르게 처리하세요.

## ZeRO 단계 설정하기[[select-a-zero-stage]]

DeepSpeed를 설치하고 메모리 요구 사항을 더 잘 파악했다면 다음 단계는 사용할 ZeRO 스테이지를 선택하는 것입니다. 가장 빠르고 메모리 효율이 높은 순서대로 정렬하면 다음과 같습니다:

| 속도              | 메모리 효율         |
|------------------|------------------|
| ZeRO-1           | ZeRO-3 + offload |
| ZeRO-2           | ZeRO-3           |
| ZeRO-2 + offload | ZeRO-2 + offload |
| ZeRO-3           | ZeRO-2           |
| ZeRO-3 + offload | ZeRO-1           |

자신에게 가장 적합한 방법을 찾으려면 가장 빠른 방법부터 시작하고 메모리가 부족하면 더 느리지만 메모리 효율이 높은 다음 단계를 시도하세요. 속도와 메모리 사용량 사이의 적절한 균형을 찾기 위해 (가장 메모리 효율적이거나 가장 빠른 것부터 시작하여) 원하는 방향으로 자유롭게 작업하세요.

일반적으로 사용할 수 있는 프로세스는 다음과 같습니다(배치 크기 1로 시작):

1. 그레이디언트 체크포인팅 활성화
2. ZeRO-2 시도
3. ZeRO-2와 매개변수 오프로드 시도
4. ZeRO-3 시도
5. ZeRO-3과 매개변수 CPU 오프로드 시도
6. ZeRO-3, 매개변수와 옵티마이저 CPU 오프로드 시도
7. [`~GenerationMixin.generate`] 메소드를 사용하는 경우 더 좁은 빔 서치 검색 범위와 같은 다양한 기본값을 낮춰보기
8. 전체 정밀도 가중치보다 반정밀도(구형 GPU 구조의 경우 fp16, 암페어 이후 GPU의 경우 bf16)를 혼합해보기
9. 가능하면 하드웨어를 더 추가하거나 Infinity가 매개변수와 옵티마이저를 NVMe로 오프로드하도록 활성화
10. 메모리가 부족하지 않으면 유효 처리량을 측정한 다음 배치 크기를 최대한 크게 늘려 GPU 효율성을 극대화
11. 마지막으로 일부 오프로드 기능을 비활성화하거나 더 빠른 ZeRO 스테이지를 사용하고 배치 크기를 늘리거나 줄여 속도와 메모리 사용량 간의 최적의 균형을 찾아 트레이닝 설정을 최적화


## DeepSpeed 구성 파일[[deepspeed-configuration-file]]

DeepSpeed는 트레이닝 실행 방법을 구성하는 모든 매개변수가 포함된 구성 파일을 통해 [`Trainer`] 클래스와 함께 작동합니다. 트레이닝 스크립트를 실행하면 DeepSpeed는 [`Trainer`]로부터 받은 구성을 콘솔에 기록하므로 어떤 구성이 사용되었는지 정확히 확인할 수 있습니다.

<Tip>

DeepSpeed 구성 옵션의 전체 목록은 [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/)에서 확인할 수 있습니다. 또한 [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) 리포지토리 또는 기본 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 리포지토리에서 다양한 DeepSpeed 구성 예제에 대한 보다 실용적인 예제를 찾을 수 있습니다. 구체적인 예제를 빠르게 찾으려면 다음과 같이 하세요:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# Lamb 옵티마이저 샘플 찾기
grep -i Lamb $(find . -name '*json')
```

</Tip>

명령줄 인터페이스에서 트레이닝하는 경우 DeepSpeed 구성 파일은 JSON 파일의 경로로 전달되거나 노트북 설정에서 [`Trainer`]를 사용하는 경우 중첩된 `dict` 객체로 전달됩니다.

<hfoptions id="pass-config">
<hfoption id="path to file">

```py
TrainingArguments(..., deepspeed="path/to/deepspeed_config.json")
```

</hfoption>
<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(..., deepspeed=ds_config_dict)
trainer = Trainer(model, args, ...)
```

</hfoption>
</hfoptions>

### DeepSpeed와 Trainer 매개변수[[deepspeed-and-trainer-parameters]]

구성 매개변수에는 세 가지 유형이 있습니다:

1. 일부 구성 매개변수는 [`Trainer`]와 DeepSpeed가 공유하며, 정의가 충돌하는 경우 오류를 식별하기 어려울 수 있습니다. 이러한 공유 구성 매개변수는 [`Trainer`] 명령줄 인수에서 쉽게 설정할 수 있습니다.

2. 모델 설정에서 자동으로 도출되는 일부 설정 매개변수는 수동으로 값을 조정할 필요가 없습니다. [`Trainer`]는 구성 값 `auto`를 사용하여 가장 정확하거나 효율적인 값을 설정합니다. 직접 구성 매개변수를 명시적으로 설정할 수도 있지만, [`Trainer`] 인수와 DeepSpeed 설정 매개변수가 일치하도록 주의해야 합니다. 일치하지 않으면 감지하기 매우 어려운 방식으로 훈련이 실패할 수 있습니다!

3. 교육 요구 사항에 따라 수동으로 설정해야 하는 일부 설정 매개변수는 DeepSpeed에만 해당됩니다.

DeepSpeed 구성을 수정하고 [`TrainingArguments`]를 편집할 수도 있습니다:

1. 기본 구성으로 사용할 DeepSpeed 구성 파일을 생성하거나 로드합니다.
2. 다음 DeepSpeed 구성을 기반으로 [`TrainingArguments`] 객체를 생성합니다.

`scheduler.params.total_num_steps`와 같은 일부 값은 트레이닝 중 [`Trainer`]에 의해 계산됩니다.

### ZeRO 구성[[zero-configuration]]

세 가지 구성이 있으며, 각 구성은 서로 다른 ZeRO 단계에 해당합니다. 1단계는 확장성 측면에서 그다지 눈여겨볼만하지 않으므로 이 가이드에서는 2단계와 3단계에 중점을 둡니다. `zero_optimization` 구성에는 활성화할 항목과 구성 방법에 대한 모든 옵션이 포함되어 있습니다. 각 매개변수에 대한 자세한 설명은 [DeepSpeed 구성 JSON](https://www.deepspeed.ai/docs/config-json/) 참조를 참조하세요.

<Tip warning={true}>
DeepSpeed는 매개변수 이름의 유효성을 검사하지 않으며 오타가 있으면 매개변수의 기본 설정으로 대체합니다. DeepSpeed 엔진 시작 로그 메시지를 보고 어떤 값을 사용할지 확인할 수 있습니다.

</Tip>

[`Trainer`]는 동등한 명령줄 인수를 제공하지 않으므로 다음 구성은 DeepSpeed로 설정해야 합니다.

<hfoptions id="zero-config">
<hfoption id="ZeRO-1">

ZeRO-1은 옵티마이저 상태를 GPU에 분할하여 약간의 속도 향상을 기대할 수 있습니다. ZeRO-1 구성은 다음과 같이 설정할 수 있습니다:

```yml
{
    "zero_optimization": {
        "stage": 1
    }
}
```

</hfoption>
<hfoption id="ZeRO-2">

ZeRO-2는 GPU에서 옵티마이저와 그레이디언트를 분할합니다. 이 단계는 추론과 관련이 없는 기능이기 때문에 주로 훈련에 사용됩니다. 더 나은 성능을 위해 구성해야 할 몇 가지 중요한 매개변수는 다음과 같습니다:

* GPU 메모리 사용량을 줄이려면 `offload_optimizer`를 활성화해야 합니다.
* `true`로 설정된 경우 `overlap_comm`은 GPU 메모리 사용량 증가를 상쇄하여 지연 시간을 줄입니다. 이 기능은 4.5배의 `allgather_bucket_size` 및 `reduce_bucket_size`값을 사용합니다. 이 예에서는 `5e8`로 설정되어 있으므로 9GB의 GPU 메모리가 필요합니다. GPU 메모리가 8GB 이하인 경우, 메모리 요구량을 낮추고 메모리 부족(OOM) 오류를 방지하기 위해 `overlap_comm`을 줄여야 합니다.
* `allgather_bucket_size`와 `reduce_bucket_size`는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다.
* DeepSpeed 0.4.4에서는 CPU 오프로딩을 위해 `round_robin_gradients`를 사용할 수 있습니다. 이 기능은 세분화된 그레이디언트 파티셔닝을 통해 등급 간 그레이디언트 복사를 CPU 메모리로 병렬화합니다. 성능 이점은 그레이디언트 누적 단계(최적화 단계 간 복사 횟수 증가) 또는 GPU 수(병렬 처리 증가)에 따라 증가합니다.

```yml
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
        "round_robin_gradients": true
    }
}
```

</hfoption>
<hfoption id="ZeRO-3">

ZeRO-3는 옵티마이저, 그래디언트, 매개변수를 여러 GPU에 걸쳐 분할합니다. ZeRO-2와 달리 ZeRO-3는 여러 GPU에 대규모 모델을 가져올 수 있기 때문에 훈련 외에도 추론에도 사용할 수 있습니다. 구성해야 할 몇 가지 중요한 매개변수는 다음과 같습니다:

* `device: "cpu"` 는 GPU 메모리가 부족하고 사용 가능한 CPU 메모리가 있는 경우 도움이 될 수 있습니다. 이를 통해 모델 매개변수를 CPU로 오프로드할 수 있습니다.
* `pin_memory: true` 는 처리량을 향상시킬 수 있지만, 핀 메모리는 메모리를 요청한 특정 프로세스를 위해 예약되어 있고 일반적으로 일반 CPU 메모리보다 훨씬 빠르게 액세스되기 때문에 다른 프로세스에서 사용할 수 있는 메모리가 줄어듭니다.
* `stage3_max_live_parameters` 는 특정 시간에 GPU에 유지하려는 전체 매개변수의 상한값입니다. OOM 오류가 발생하면 이 값을 줄이세요.
* `stage3_max_reuse_distance` 는 향후 매개변수를 다시 사용할 시기를 결정하는 값으로, 매개변수를 버릴지 유지할지 결정하는 데 도움이 됩니다. 매개변수를 재사용할 경우(`stage3_max_reuse_distance`보다 작은 값인 경우) 통신 오버헤드를 줄이기 위해 매개변수를 유지합니다. 이 기능은 활성화 체크포인팅이 활성화되어 있고 역전파 계산시까지 순전파 시점의 매개변수를 유지하려는 경우에 매우 유용합니다. 그러나 OOM 오류가 발생하면 이 값을 줄이세요.
* 모델 저장 시 `stage3_gather_16bit_weights_on_model_save`는 fp16 가중치를 통합합니다. 대규모 모델을 학습하거나 여러 GPU를 사용할 경우 메모리와 속도 측면에서 비용이 많이 듭니다. 훈련을 재개할 계획이라면 이 옵션을 활성화해야 합니다.
* `sub_group_size` 는 최적화 단계에서 업데이트되는 매개변수를 제어합니다. 매개변수는 `sub_group_size`의 버킷으로 그룹화되며 각 버킷은 한 번에 하나씩 업데이트됩니다. NVMe 오프로드와 함께 사용하는 경우 `sub_group_size`는 최적화 단계 중 모델 상태가 CPU 메모리로 이동하는 시점을 결정합니다. 이렇게 하면 매우 큰 모델의 CPU 메모리 부족을 방지할 수 있습니다. NVMe 오프로드를 사용하지 않는 경우 `sub_group_size`를 기본값으로 둘 수 있지만, 사용하는 경우 변경하는 것이 좋습니다:

    1. 옵티마이저 단계에서 OOM 오류가 발생합니다. 이 경우, 임시 버퍼의 메모리 사용량을 줄이려면 `sub_group_size`를 줄이세요.
    2. 옵티마이저 단계에서 시간이 너무 오래 걸립니다. 이 경우 데이터 버퍼 증가로 인한 대역폭 사용률을 개선하기 위해 `sub_group_size`를 늘리세요.

* `reduce_bucket_size`, `stage3_prefetch_bucket_size`, `stage3_param_persistence_threshold`는 모델의 숨겨진 크기에 따라 달라집니다. 이 값들을 `auto`으로 설정하고 [`Trainer`]가 자동으로 값을 할당하도록 허용하는 것이 좋습니다.

```yml
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

[`deepspeed.zero.Init`](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.Init) 컨텍스트 매니저를 사용하면 모델을 더 빠르게 초기화할 수 있습니다:

```py
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
    config = T5Config.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration(config)
```

사전 학습된 모델의 경우, 딥스피드 구성 파일에 `is_deepspeed_zero3_enabled: true`가 [`TrainingArguments`]에 설정되어 있어야 하며, ZeRO 구성이 활성화되어 있어야 합니다. 훈련된 모델 [`~PreTrainedModel.from_pretrained`]을 호출하기 **전에** [`TrainingArguments`] 객체를 생성해야 합니다.

```py
from transformers import AutoModel, Trainer, TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model, args=training_args, ...)
```

fp16 가중치가 단일 GPU에 맞지 않는 경우 ZeRO-3이 필요합니다. fp16 가중치를 로드할 수 있는 경우, [`~PreTrainedModel.from_pretrained`]에 `torch_dtype=torch.float16`을 지정해야 합니다.

ZeRO-3의 또 다른 고려 사항은 여러 개의 GPU를 사용하는 경우 현재 실행 중인 레이어의 매개변수가 아닌 한 단일 GPU에 모든 매개변수가 없다는 것입니다. 사전 훈련된 모델 가중치를 [`~PreTrainedModel.from_pretrained`]에 로드하는 등 모든 레이어의 모든 매개변수에 한 번에 액세스하려면 한 번에 하나의 레이어를 로드하고 즉시 모든 GPU에 파티셔닝합니다. 이는 매우 큰 모델의 경우 메모리 제한으로 인해 하나의 GPU에 가중치를 로드한 다음 다른 GPU에 분산할 수 없기 때문입니다.

다음과 같이 보이는 모델 매개변수 가중치(여기서 `tensor([1.])`) 또는 매개변수 크기가 더 큰 다차원 형태 대신 1인 경우, 이는 매개변수가 분할되어 있으며 이것이 ZeRO-3 플레이스홀더인 것을 의미합니다.

```py
tensor([1.0], device="cuda:0", dtype=torch.float16, requires_grad=True)
```

<Tip>

ZeRO-3로 대규모 모델을 초기화하고 매개변수에 액세스하는 방법에 대한 자세한 내용은 [Constructing Massive Models](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) 및 [Gathering Parameters](https://deepspeed.readthedocs.io/en/latest/zero3.html#gathering-parameters) 가이드를 참조하세요.

</Tip>

</hfoption>
</hfoptions>

### NVMe 설정[[nvme-configuration]]

[ZeRO-Infinity](https://hf.co/papers/2104.07857)를 사용하면 모델 상태를 CPU 및/또는 NVMe로 오프로드하여 더 많은 메모리를 절약할 수 있습니다. 스마트 파티셔닝 및 타일링 알고리즘을 통해 각 GPU는 오프로딩 중에 매우 적은 양의 데이터를 주고받을 수 있으므로 최신 NVMe는 훈련 프로세스에 사용할 수 있는 것보다 훨씬 더 큰 총 메모리 풀에 맞출 수 있습니다. ZeRO-Infinity에는 ZeRO-3가 필요합니다.

사용 가능한 CPU 및/또는 NVMe 메모리에 따라 [옵티마이저](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading)와 [매개변수](https://www.deepspeed.ai/docs/config-json/#parameter-offloading) 중 하나만 오프로드하거나 아무것도 오프로드하지 않을 수 있습니다. 또한 일반 하드 드라이브나 솔리드 스테이트 드라이브에서도 작동하지만 속도가 현저히 느려지므로 `nvme_path`가 NVMe 장치를 가리키고 있는지 확인해야 합니다. 최신 NVMe를 사용하면 읽기 작업의 경우 최대 3.5GB/s, 쓰기 작업의 경우 최대 3GB/s의 전송 속도를 기대할 수 있습니다. 마지막으로, 트레이닝 설정에서 [벤치마크 실행하기](https://github.com/microsoft/DeepSpeed/issues/998)을 통해 최적의 'aio' 구성을 결정합니다.

아래 예제 ZeRO-3/Infinity 구성 파일은 대부분의 매개변수 값을 `auto`으로 설정하고 있지만, 수동으로 값을 추가할 수도 있습니다.

```yml
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

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

## DeepSpeed 구성[[deepspeed-features]]

이 섹션에서 간략하게 설명하는 몇 가지 중요한 매개변수를 DeepSpeed 구성 파일에 지정할 수 있습니다.

### 활성화/그레이디언트 체크포인팅[[activationgradient-checkpointing]]

활성화 및 그레이디언트 체크포인팅은 속도를 더 많은 GPU 메모리와 교환하여 GPU 메모리가 부족한 상황을 극복하거나 배치 크기를 늘려 성능을 향상시킬 수 있습니다. 이 기능을 활성화하려면 다음과 같이 하세요:

1. 허깅 페이스 모델의 경우, [`Trainer`]에서 `model.gradient_checkpointing_enable()` 또는 `--gradient_checkpointing`을 설정합니다.
2. 허깅 페이스가 아닌 모델의 경우, 딥스피드 [Activation Checkpointing API](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html)를 사용합니다. 트랜스포머 모델링 코드를 대체하고 `torch.utils.checkpoint`를 DeepSpeed API로 대체할 수도 있습니다. 이 접근 방식은 순방향 활성화를 다시 계산하는 대신 CPU 메모리로 오프로드할 수 있으므로 더 유연합니다.

### 옵티마이저와 스케줄러[[optimizer-and-scheduler]]

`offload_optimizer`를 활성화하지 않는 한 DeepSpeed와 트랜스포머 옵티마이저 및 스케줄러를 혼합하여 사용할 수 있습니다. `offload_optimizer`를 활성화하면 CPU와 GPU 구현이 모두 있는 경우 DeepSpeed가 아닌 최적화기(LAMB 제외)를 사용할 수 있습니다.

<Tip warning={true}>

구성 파일의 최적화 프로그램 및 스케줄러 매개변수는 명령줄에서 설정할 수 있으므로 오류를 찾기 어렵지 않습니다. 예를 들어 학습 속도가 다른 곳에서 다른 값으로 설정된 경우 명령줄에서 이를 재정의할 수 있습니다. 최적화 프로그램 및 스케줄러 매개변수 외에도 [`Trainer`] 명령줄 인수가 DeepSpeed 구성과 일치하는지 확인해야 합니다.

</Tip>

<hfoptions id="opt-sched">
<hfoption id="optimizer">

DeepSpeed는 여러 [옵티마이저](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)를 제공하지만(Adam, AdamW, OneBitAdam 및 LAMB) PyTorch에서 다른 옵티마이저를 가져올 수도 있습니다. 설정에서 옵티마이저를 구성하지 않으면 [`Trainer`]가 자동으로 AdamW를 선택하고 명령줄에서 제공된 값 또는 기본값을 사용합니다: `lr`, `adam_beta1`, `adam_beta2`, `adam_epsilon`, `weight_decay`.

매개변수를 `"auto"`으로 설정하거나 원하는 값을 직접 수동으로 입력할 수 있습니다.

```yaml
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

최상위 구성에 다음을 추가하여 지원되지 않는 옵티마이저를 사용할 수도 있습니다.

```yaml
{
   "zero_allow_untested_optimizer": true
}
```

DeepSpeed==0.8.3부터 오프로드를 사용하려면 오프로드가 DeepSpeed의 CPU Adam 옵티마이저에서 가장 잘 작동하므로 최상위 수준 구성에 다음 사항을 추가해야 합니다.

```yaml
{
   "zero_force_ds_cpu_optimizer": false
}
```

</hfoption>
<hfoption id="scheduler">

DeepSpeed는 LRRangeTest, OneCycle, WarmupLR 및 WarmupDecayLR learning rate[schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters)를 지원합니다.

트랜스포머와 DeepSpeed는 동일한 두 가지 스케줄러를 제공합니다:

* WarmupLR은 Transformers의 `--lr_scheduler_type constant_warmup`과 동일합니다.
* WarmupDecayLR은 Transformers의 `--lr_scheduler_type linear`와 동일합니다(Transformers에서 사용되는 기본 스케줄러입니다).

설정에서 스케줄러를 구성하지 않으면[`Trainer`]는 자동으로 WarmupDecayLR을 선택하고 명령줄에서 제공된 값 또는 기본값을 사용합니다: `warmup_min_lr`, `warmup_max_lr`, `warmup_num_steps`, `total_num_steps` (`max_steps`가 제공되지 않으면 런타임 중에 자동으로 계산됨).

매개변수를 `"auto"`으로 설정하거나 원하는 값을 직접 수동으로 입력할 수 있습니다.

```yaml
{
   "scheduler": {
         "type": "WarmupDecayLR",
         "params": {
             "total_num_steps": "auto",
             "warmup_min_lr": "auto",
             "warmup_max_lr": "auto",
             "warmup_num_steps": "auto"
         }
     }
}
```

</hfoption>
</hfoptions>

### 정밀도[[precision]]

DeepSpeed는 fp32, fp16 및 bf16 혼합 정밀도를 지원합니다.

<hfoptions id="precision">
<hfoption id="fp32">

모델이 혼합 정밀도로 사전 학습되지 않은 경우와 같이 혼합 정밀도로 잘 작동하지 않는 경우 NaN 손실을 유발할 수 있는 오버플로 또는 언더플로 문제가 발생할 수 있습니다. 이러한 경우에는 기본 fp16 모드를 명시적으로 비활성화하여 전체 fp32 정밀도를 사용해야 합니다.

```yaml
{
    "fp16": {
        "enabled": false
    }
}
```

Ampere GPU 및 PyTorch 1.7 이상의 경우 일부 연산에 대해 더 효율적인 [tf32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) 형식으로 자동 전환되지만 결과는 여전히 fp32로 표시됩니다. [`Trainer`]에서 `--tf32`를 설정하여 활성화하고 `--tf32 0` 또는 `--no_tf32`를 비활성화하면 제어할 수 있습니다.

</hfoption>
<hfoption id="fp16">

PyTorch AMP와 같은 fp16 혼합 정밀도를 구성하면 메모리 사용량이 줄어들고 훈련 속도가 빨라집니다.[`Trainer`]는 `args.fp16_backend` 값에 따라 fp16을 자동으로 활성화 또는 비활성화하며, 나머지 구성은 사용자가 설정할 수 있습니다. 명령줄에서 다음 인수를 전달하면 fp16이 활성화됩니다: `fp16`, `--fp16_backend amp` 또는 `--fp16_full_eval`.

```yaml
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

추가 딥스피드 fp16 훈련 옵션은 [fp16 훈련 옵션](https://www.deepspeed.ai/docs/config-json/#fp16-training-options) 참조를 참조하세요.

Apex와 같은 fp16 혼합 정밀도를 구성하려면 아래 그림과 같이 `"auto"` 또는 직접 값을 설정합니다.[`Trainer`]는 `args.fp16_backend` 및 `args.fp16_opt_level`의 값에 따라 `amp`를 자동으로 구성합니다. 다음 인수를 전달하면 명령줄에서 활성화할 수도 있습니다: `fp16`, `--fp16_backend apex` 또는 `--fp16_opt_level 01`.

```yaml
{
    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }
}
```

</hfoption>
<hfoption id="bf16">

bf16을 사용하려면 DeepSpeed==0.6.0 이상이 필요합니다. bf16은 fp32와 동적 범위가 동일하며 손실 스케일링이 필요하지 않습니다. 그러나 [gradient accumulation](#gradient-accumulation)을 bf16과 함께 사용하면 이 형식의 낮은 정밀도로 인해 손실이 발생할 수 있으므로 원하지 않는 그레이디언트가 bf16에 누적될 수 있습니다.

bf16은 설정 파일에서 설정하거나 다음 인수를 전달하면 명령줄에서 활성화할 수 있습니다: `--bf16` 또는 `--bf16_full_eval`.

```yaml
{
    "bf16": {
        "enabled": "auto"
    }
}
```

</hfoption>
</hfoptions>

### 배치 크기[[batch-size]]

배치 크기는 자동으로 구성하거나 명시적으로 설정할 수 있습니다. `"auto"` 옵션을 사용하도록 선택하면 [`Trainer`]는 `train_micro_batch_size_per_gpu`를 args.`per_device_train_batch_size`의 값으로, `train_batch_size`를 `args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`로 설정합니다.

```yaml
{
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto"
}
```

### 그레이디언트 누적[[gradient-accumulation]]

그레이디언트 누적을 자동으로 구성하거나 명시적으로 설정할 수 있습니다. `"auto"` 옵션을 사용하도록 선택하면 [`Trainer`]가 `args.gradient_accumulation_steps`의 값으로 설정합니다.

```yaml
{
    "gradient_accumulation_steps": "auto"
}

```

### 그레이디언트 클리핑[[gradient-clipping]]

그레이디언트 클리핑은 자동으로 구성하거나 명시적으로 설정할 수 있습니다. `"auto"` 옵션을 사용하도록 선택하면 [`Trainer`]가 `args.max_grad_norm`의 값으로 설정합니다.

```yaml
{
    "gradient_clipping": "auto"
}
```

### 통신 데이터 유형(Communication data type)[[communication-data-type]]

축소, 수집 및 분산 작업과 같은 통신 집합체의 경우 별도의 데이터 유형이 사용됩니다.

모든 수집 및 분산 작업은 데이터와 동일한 데이터 유형으로 수행됩니다. 예를 들어 bf16으로 훈련하는 경우, 수집은 비손실 연산이므로 데이터도 bf16으로 수집됩니다.

예를 들어 그레이디언트가 여러 GPU에 걸쳐 평균화되는 경우와 같이 감소 연산은 손실이 발생합니다. 통신이 fp16 또는 bf16으로 수행되는 경우, 낮은 정밀도로 여러 숫자를 더하면 정확하지 않기 때문에 손실이 발생할 가능성이 더 높습니다. 특히 fp16보다 정밀도가 낮은 bf16의 경우 더욱 그렇습니다. 이러한 이유로 기울기를 평균화할 때 손실이 최소화되므로 감소 연산에는 fp16이 기본값으로 사용됩니다.

통신 데이터 유형은 설정 파일에서 `communication_data_type` 매개변수를 설정하여 선택할 수 있습니다. 예를 들어, fp32를 선택하면 약간의 오버헤드가 추가되지만 감소 연산이 fp32에 누적되고 준비가 되면 훈련 중인 반정밀 dtype으로 다운캐스트됩니다.

```yaml
{
    "communication_data_type": "fp32"
}
```

## 모델 배포[[deployment]]

[torchrun](https://pytorch.org/docs/stable/elastic/run.html), `deepspeed` 런처 또는 [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch) 등 다양한 런처를 통해 DeepSpeed를 배포할 수 있습니다. 배포하려면 [`Trainer`] 명령줄에 `--deepspeed ds_config.json`을 추가합니다. 필요한 명령줄 인수를 코드에 추가하려면 DeepSpeed의 [`add_config_arguments`](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) 유틸리티를 사용하는 것이 좋습니다.

이 가이드에서는 다양한 트레이닝 설정에 대해 `deepspeed` 런처로 DeepSpeed를 배포하는 방법을 보여드립니다. 보다 실용적인 사용 예제는 이 [post](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400)에서 확인할 수 있습니다.

<hfoptions id="deploy">
<hfoption id="multi-GPU">

여러 GPU에 DeepSpeed를 배포하려면 `--num_gpus` 매개변수를 추가하세요. 사용 가능한 모든 GPU를 사용하려는 경우 `--num_gpus`를 추가할 필요가 없습니다. 아래 예제에서는 2개의 GPU를 사용합니다.

```bash
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

</hfoption>
<hfoption id="single-GPU">

단일 GPU에 DeepSpeed를 배포하려면 `--num_gpus` 매개변수를 추가하세요. GPU가 1개만 있는 경우 이 값을 명시적으로 설정할 필요는 없습니다. DeepSpeed는 지정된 노드에서 볼 수 있는 모든 GPU를 배포하므로 이 값을 명시적으로 설정할 필요는 없습니다.

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

DeepSpeed는 단 하나의 GPU로도 여전히 유용합니다:

1. 일부 계산과 메모리를 CPU로 오프로드하여 더 큰 배치 크기를 사용하거나 일반적으로 맞지 않는 매우 큰 모델을 맞추기 위해 모델에 더 많은 GPU 리소스를 사용할 수 있도록 합니다.
2. 스마트 GPU 메모리 관리 시스템으로 메모리 조각화를 최소화하여 더 큰 모델과 데이터 배치에 맞출 수 있습니다.

<Tip>

단일 GPU에서 더 나은 성능을 얻으려면 [ZeRO-2](#zero-configuration) 구성 파일에서 `allgather_bucket_size` 및 `reduce_bucket_size` 값을 2e8로 설정하세요.

</Tip>

</hfoption>
</hfoptions>

### 다중 노드 환경에서의 모델 배포[[multi-node-deployment]]

노드는 워크로드를 실행하기 위한 하나 이상의 GPU입니다. 더 강력한 설정은 멀티 노드 설정으로, `deepspeed` 런처로 실행할 수 있습니다. 이 가이드에서는 각각 8개의 GPU가 있는 두 개의 노드가 있다고 가정해 보겠습니다. 첫 번째 노드는 `ssh hostname1`로, 두 번째 노드는 `ssh hostname2`로 접속할 수 있습니다. 두 노드 모두 비밀번호 없이 ssh를 통해 로컬로 서로 통신할 수 있어야 합니다.

기본적으로 DeepSpeed는 멀티노드 환경에서 공유 저장소를 사용할 것으로 예상합니다. 그렇지 않고 각 노드가 로컬 파일 시스템만 볼 수 있는 경우, 공유 파일 시스템에 대한 액세스 없이 로딩할 수 있도록 [`checkpoint`](https://www.deepspeed.ai/docs/config-json/#checkpoint-options)를 포함하도록 구성 파일을 조정해야 합니다:

```yaml
{
  "checkpoint": {
    "use_node_local_storage": true
  }
}
```

[`Trainer`]의 ``--save_on_each_node` 인수를 사용하여 위의 `checkpoint`를 구성에 자동으로 추가할 수도 있습니다.

<hfoptions id="multinode">
<hfoption id="torchrun">

[torchrun](https://pytorch.org/docs/stable/elastic/run.html)의 경우, 각 노드에 ssh로 접속한 후 두 노드 모두에서 다음 명령을 실행해야 합니다. 런처는 두 노드가 동기화될 때까지 기다렸다가 트레이닝을 시작합니다.

```bash
torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

</hfoption>
<hfoption id="deepspeed">

`deepspeed` 런처의 경우, 먼저 `hostfile`을 생성합니다.

```bash
hostname1 slots=8
hostname2 slots=8
```

그런 다음 다음 명령어로 트레이닝을 시작할 수 있습니다. `deepspeed` 런처는 두 노드에서 동시에 명령을 자동으로 실행합니다.

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

다중 노드 컴퓨팅 리소스 구성에 대한 자세한 내용은 [Resource Configuration (multi-node)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) 가이드를 참조하세요.

</hfoption>
</hfoptions>

### SLURM[[slurm]]

SLURM 환경에서는 특정 SLURM 환경에 맞게 SLURM 스크립트를 조정해야 합니다.SLURM 스크립트 예시는 다음과 같습니다:

```bash
#SBATCH --job-name=test-nodes        # 작업 이름
#SBATCH --nodes=2                    # 노드 수
#SBATCH --ntasks-per-node=1          # 중요 - 노드당 분산 작업 1개!
#SBATCH --cpus-per-task=10           # 작업당 CPU 코어 수
#SBATCH --gres=gpu:8                 # gpu 수
#SBATCH --time 20:00:00              # 최대 실행 시간 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 출력 파일 이름

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

그런 다음 모든 노드에서 동시에 학습을 시작하는 다음 명령을 사용하여 다중 노드 배포를 예약할 수 있습니다.

```bash
sbatch launch.slurm
```

### 노트북[[notebook]]

`deepspeed` 런처는 노트북에서의 배포를 지원하지 않으므로 분산 환경을 에뮬레이션해야 합니다. 하지만 이는 1개의 GPU에서만 작동합니다. 1개 이상의 GPU를 사용하려면 딥스피드가 작동할 수 있는 다중 프로세스 환경을 사용해야 합니다. 즉, 여기에 표시된 것처럼 에뮬레이션할 수 없는 `deepspeed` 런처를 사용해야 합니다.

```py
# DeepSpeed는 단일 프로세스만 사용하더라도 분산 환경을 필요로 합니다.
# 이 코드로 분산 환경을 모방합니다.
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # RuntimeError: Address already in use 오류 발생 시 수정
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# 이제 평소와 같이 진행하되, DeepSpeed 설정 파일을 전달합니다.
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

현재 디렉터리의 노트북에 구성 파일을 즉석에서 만들고 싶다면 전용 셀을 만들 수 있습니다.

```py
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

트레이닝 스크립트가 노트북 셀이 아닌 파일에 있는 경우, 노트북 셀의 셸에서 `deepspeed`를 정상적으로 실행할 수 있습니다. 예를 들어 `run_translation.py`를 시작하려면 다음과 같이 하세요.:

```py
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

또한 `%%bash` 매직을 사용하여 여러 줄의 코드를 작성하여 셸 프로그램을 실행할 수도 있지만 교육이 완료될 때까지 로그를 볼 수 없습니다. `%%bash` 매직으로 분산 환경을 에뮬레이션할 필요는 없습니다.

```py
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```

## 모델 가중치 저장하기[[save-model-weights]]

딥스피드는 기본 고정밀 fp32 가중치를 사용자 지정 체크포인트 최적화 파일(glob 패턴은 `global_step*/*optim_states.pt`처럼 보입니다)에 저장하고 일반 체크포인트 아래에 저장합니다.

<hfoptions id="save">
<hfoption id="fp16">

ZeRO-2로 훈련된 모델은 pytorch_model.bin 가중치를 fp16에 저장합니다. ZeRO-3으로 훈련된 모델의 모델 가중치를 fp16에 저장하려면 모델 가중치가 여러 GPU에 분할되어 있으므로 `“stage3_gather_16bit_weights_on_model_save”: true`를 설정해야 합니다. 그렇지 않으면 [`Trainer`]가 가중치를 fp16에 저장하지 않고 pytorch_model.bin 파일을 생성하지 않습니다. 이는 DeepSpeed의 state_dict에 실제 가중치 대신 플레이스홀더가 포함되어 있어 이를 로드할 수 없기 때문입니다.

```yaml
{
    "zero_optimization": {
        "stage3_gather_16bit_weights_on_model_save": true
    }
}
```

</hfoption>
<hfoption id="fp32">

전체 정밀 가중치는 많은 메모리가 필요할 수 있으므로 트레이닝 중에 저장해서는 안 됩니다. 일반적으로 훈련이 완료된 후 오프라인으로 fp32 가중치를 저장하는 것이 가장 좋습니다. 그러나 여유 CPU 메모리가 많은 경우 훈련 중에 fp32 가중치를 저장할 수 있습니다. 이 섹션에서는 온라인과 오프라인 방식을 모두 다룹니다.

### 온라인 환경[[online]]

다음과 같이 최신 체크포인트를 로드하려면 체크포인트를 하나 이상 저장해야 합니다:

```py
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

`--load_best_model_at_end` 매개변수를 활성화하여 [`TrainingArguments`]에서 최적의 체크포인트를 추적하는 경우, 먼저 학습을 완료하고 최종 모델을 명시적으로 저장할 수 있습니다. 그런 다음 아래와 같이 다시 로드할 수 있습니다:

```py
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
trainer.deepspeed.save_checkpoint(checkpoint_dir)
fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
```

<Tip>

`load_state_dict_from_zero_checkpoint`가 실행되면 동일한 애플리케이션의 컨텍스트에서 모델을 더 이상 DeepSpeed에서 사용할 수 없습니다. `model.load_state_dict(state_dict)`는 모든 딥스피드 마법을 제거하므로 딥스피드 엔진을 다시 초기화해야 합니다. 이 기능은 훈련이 끝날 때만 사용하세요.

</Tip>

fp32 가중치의 state_dict를 추출하여 로드할 수도 있습니다:

```py
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)  # cpu에 이미 존재함
model = model.cpu()
model.load_state_dict(state_dict)
```

### 오프라인 환경[[offline]]

DeepSpeed는 언제든지 가중치를 추출할 수 있도록 체크포인트 폴더의 최상위 레벨에 zero_to_fp32.py 스크립트를 제공합니다. 이 스크립트는 독립형 스크립트로 구성 파일이나 [`Trainer`]가 필요하지 않습니다.

예를 들어 체크포인트 폴더가 다음과 같은 경우입니다:

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

딥스피드 체크포인트(ZeRO-2 또는 ZeRO-3) 하위 폴더 `global_step1`에서 fp32 가중치를 재구성하려면 다음 명령을 실행하여 여러 GPU의 전체 fp32 가중치를 단일 pytorch_model.bin 파일로 생성하고 통합합니다. 스크립트는 자동으로 체크포인트가 포함된 하위 폴더를 찾습니다.

```py
python zero_to_fp32.py . pytorch_model.bin
```

<Tip>

자세한 사용법은 `python zero_to_fp32.py -h`를 실행하세요. 이 스크립트에는 최종 fp32 가중치의 2배의 일반 RAM이 필요합니다.

</Tip>

</hfoption>
</hfoptions>

## ZeRO Inference[[zero-inference]]

[ZeRO Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)는 모델 가중치를 CPU 또는 NVMe 메모리에 배치하여 GPU에 부담을 주지 않으므로 GPU에서 대규모 모델을 사용하여 추론을 실행할 수 있습니다. 추론은 최적화 상태 및 그레이디언트에 많은 양의 메모리를 추가로 필요로 하지 않으므로 동일한 하드웨어에 훨씬 더 큰 배치 및/또는 시퀀스 길이를 맞출 수 있습니다.

ZeRO Inference는 [ZeRO-3](#zero-configuration)와 동일한 구성 파일을 공유하며, ZeRO-2 및 ZeRO-1 구성은 추론에 아무런 이점을 제공하지 않으므로 작동하지 않습니다.

ZeRO Inference를 실행하려면 일반적인 훈련 인수를 [`TrainingArguments`] 클래스에 전달하고 `--do_eval` 인수를 추가합니다.

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json
```

## Trainer 없이 DeepSpeed 사용하기[[non-trainer-deepspeed-integration]]

DeepSpeed는 [`Trainer`] 클래스가 없는 트랜스포머에서도 작동합니다. 이는 [`~PreTrainedModel.from_pretrained`]를 호출할 때 ZeRO-3 매개변수를 수집하고 모델을 여러 GPU에 분할하는 작업만 처리하는 [`HfDeepSpeedConfig`]가 처리합니다.

<Tip>

모든 것이 자동으로 처리되기를 원한다면, [`Trainer`]와 함께 DeepSpeed를 사용해 보세요! [DeepSpeed 문서](https://www.deepspeed.ai/)를 참조하여 설정 파일에서 매개변수 값을 수동으로 구성해야 합니다(`"auto"` 값은 사용할 수 없음).

</Tip>

ZeRO-3를 효율적으로 배포하려면 모델 앞에 [`HfDeepSpeedConfig`] 객체를 인스턴스화하고 해당 객체를 유지해야 합니다:

<hfoptions id="models">
<hfoption id="pretrained model">

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel
import deepspeed

ds_config = {...}  # deepspeed 설정 객체 또는 파일 경로
# Zero 3를 감지하기 위해 모델을 인스턴스화하기 전에 반드시 실행해야 합니다
dschf = HfDeepSpeedConfig(ds_config)  # 이 객체를 유지하세요.
model = AutoModel.from_pretrained("openai-community/gpt2")
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
<hfoption id="non-pretrained model">

[`HfDeepSpeedConfig`] is not required for ZeRO-1 or ZeRO-2.

```py
from transformers.integrations import HfDeepSpeedConfig
from transformers import AutoModel, AutoConfig
import deepspeed

ds_config = {...}  # deepspeed 설정 객체 또는 파일 경로
# Zero 3를 감지하기 위해 모델을 인스턴스화하기 전에 반드시 실행해야 합니다
dschf = HfDeepSpeedConfig(ds_config)  # 이 객체를 유지하세요.
config = AutoConfig.from_pretrained("openai-community/gpt2")
model = AutoModel.from_config(config)
engine = deepspeed.initialize(model=model, config_params=ds_config, ...)
```

</hfoption>
</hfoptions>

### Trainer 없이 ZeRO Inference 사용하기[[non-trainer-zero-inference]]

단일 GPU에 모델을 맞출 수 없는 경우 [`Trainer`]없이 ZeRO 추론을 실행하려면 추가 GPU를 사용하거나 CPU 메모리로 오프로드를 시도하세요. 여기서 이해해야 할 중요한 뉘앙스는 ZeRO가 설계된 방식에 따라 서로 다른 GPU에서 서로 다른 입력을 병렬로 처리할 수 있다는 것입니다.

반드시 확인하세요:

* GPU 메모리가 충분한 경우 CPU 오프로드를 비활성화합니다(속도가 느려지므로).
* Ampere 이상의 GPU를 사용하는 경우 bf16을 활성화하면 속도가 빨라집니다. 이러한 GPU가 없는 경우 오버플로 오류가 발생할 수 있으므로 bf16으로 사전 학습된 모델(T5 모델)을 사용하지 않는 한 fp16을 활성화할 수 있습니다.

단일 GPU에 맞지 않는 모델에서 [`Trainer`] 없이 ZeRO 추론을 실행하는 방법에 대한 더 나은 아이디어를 얻으려면 다음 스크립트를 살펴보시기 바랍니다.

```py
#!/usr/bin/env python

# 이 스크립트는 단일 GPU에 모델을 맞출 수 없을 때 추론 모드에서 Deepspeed ZeRO를 사용하는 방법을 보여줍니다.
#
# 1. CPU 오프로드와 함께 1개의 GPU 사용
# 2. 또는 여러 GPU 사용
#
# 먼저 deepspeed를 설치해야 합니다: pip install deepspeed
#
# 여기서는 약 15GB의 GPU RAM이 필요한 3B "bigscience/T0_3B" 모델을 사용합니다 - 따라서 1개의 큰 GPU나 2개의
# 작은 GPU로 처리할 수 있습니다. 또는 1개의 작은 GPU와 많은 CPU 메모리로도 가능합니다.
#
# 약 50GB가 필요한 "bigscience/T0"와 같은 더 큰 모델을 사용하려면, 80GB GPU가 없는 한
# 2-4개의 GPU가 필요할 것입니다. 그리고 여러 입력을 한 번에 처리하고 싶다면
# 스크립트를 수정하여 더 많은 GPU를 처리할 수 있습니다.
#
# 제공된 deepspeed 설정은 CPU 메모리 오프로딩도 활성화하므로, 사용 가능한 CPU 메모리가 많고
# 속도 저하를 감수할 수 있다면 일반적으로 단일 GPU에 맞지 않는 모델을 로드할 수 있을 것입니다.
# GPU 메모리가 충분하다면 CPU로의 오프로드를 원하지 않을 때 프로그램이 더 빠르게 실행될 것입니다 - 그럴 때는 해당 섹션을 비활성화하세요.
#
# 1개의 GPU에 배포하려면:
#
# deepspeed --num_gpus 1 t0.py
# 또는:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# 2개의 GPU에 배포하려면:
#
# deepspeed --num_gpus 2 t0.py
# 또는:
# python -m torch.distributed.run --nproc_per_node=2 t0.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 토크나이저의 병렬 처리에 관한 경고를 피하기 위함입니다.

# 분산 환경 설정
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

model_name = "bigscience/T0_3B"

config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.d_model

# 배치 크기는 world_size로 나누어 떨어져야 하지만, world_size보다 클 수 있습니다
train_batch_size = 1 * world_size

# ds_config 참고사항
#
# - Ampere 이상의 GPU를 사용하는 경우 bf16을 활성화하세요 - 이는 혼합 정밀도로 실행되어
# 더 빠를 것입니다.
#
# - 오래된 GPU의 경우 fp16을 활성화할 수 있지만, bf16으로 사전 훈련되지 않은 모델에서만 작동합니다 - 예를 들어
# 모든 공식 t5 모델은 bf16으로 사전 훈련되었습니다
#
# - CPU 오프로드를 원하지 않는다면 offload_param.device를 "none"으로 설정하거나 `offload_param` 섹션을
# 완전히 제거하세요
#
# - `offload_param`을 사용하는 경우, stage3_param_persistence_threshold를 수동으로 미세 조정하여
# 어떤 매개변수가 GPU에 남아있어야 하는지 제어할 수 있습니다 - 값이 클수록 오프로드 크기가 작아집니다
#
# Deepspeed 설정에 대한 자세한 정보는 다음을 참조하세요
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# 일관성을 위해 json과 동일한 형식을 유지하되, true/false에는 소문자를 사용합니다
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

# 다음 줄은 모델의 `from_pretrained` 메소드가 호출될 때
# deepspeed.zero.Init를 사용하여 모델을 여러 GPU에 직접 분할하도록 transformers에 지시합니다.
#
# **이는 AutoModelForSeq2SeqLM.from_pretrained(model_name)로 모델을 로드하기 전에 실행되어야 합니다**
#
# 그렇지 않으면 모델이 먼저 정상적으로 로드된 후 포워드 시에만 분할되는데, 이는
# 덜 효율적이며 CPU RAM이 부족할 경우 실패할 수 있습니다
dschf = HfDeepSpeedConfig(ds_config)  # 이 객체를 유지하세요

# 이제 모델을 로드할 수 있습니다.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Deepspeed ZeRO를 초기화하고 엔진 객체만 저장
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# Deepspeed ZeRO는 각 GPU에서 서로 관련 없는 입력을 처리할 수 있습니다. 따라서 2개의 GPU를 사용하면 한 번에 2개의 입력을 처리할 수 있습니다.
# GPU를 더 많이 사용하는 경우 그에 맞게 조정하세요.

# 물론 처리할 입력이 하나뿐이라면 두 GPU에 동일한 문자열을 전달해야 합니다.
# GPU를 하나만 사용하는 경우에는 rank 0만 갖게 됩니다.
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

스크립트를 t0.py로 저장하고 실행합니다:

```bash
$ deepspeed --num_gpus 2 t0.py
rank0:
   in=Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy
  out=Positive
rank1:
   in=Is this review positive or negative? Review: this is the worst restaurant ever
  out=negative
```

이것은 매우 기본적인 예시이므로 사용 사례에 맞게 조정할 수 있습니다.

### 생성[[generate]]

생성에 ZeRO-3와 함께 여러 개의 GPU를 사용하려면 [`~GenerationMixin.generate`] 메서드에서 `synced_gpus=True`를 설정하여 GPU를 동기화해야 합니다. 그렇지 않으면 한 GPU가 다른 GPU보다 먼저 생성을 완료하면 나머지 GPU가 먼저 완료한 GPU로부터 가중치 샤드를 받지 못하여 전체 시스템이 중단됩니다.

트랜스포머>=4.28의 경우, 생성 중에 여러 개의 GPU가 감지되면 `synced_gpus`가 자동으로 `True`로 설정됩니다.

## 트러블슈팅[[troubleshoot]]

문제가 발생하면 DeepSpeed가 문제의 원인이 아닌 경우가 많으므로(아주 명백하고 예외적으로 DeepSpeed 모듈을 볼 수 있는 경우가 아니라면) DeepSpeed가 문제의 원인인지 고려해야 합니다! 첫 번째 단계는 DeepSpeed 없이 설정을 다시 시도하고 문제가 지속되면 문제를 신고하는 것입니다. 문제가 핵심적인 DeepSpeed 문제이고 transformers와 관련이 없는 경우, [DeepSpeed 리포지토리](https://github.com/microsoft/DeepSpeed)에서 이슈를 개설하세요.

transformers와 관련된 이슈를 개설할 때에는 다음 정보를 제공해 주세요:

* 전체 DeepSpeed 구성 파일

*[`Trainer`]의 명령줄 인수, 또는[`Trainer`] 설정을 직접 작성하는 경우[`TrainingArguments`] 인수(관련 없는 항목이 수십 개 있는 [`TrainingArguments`]는 덤프하지 마세요).

* 다음 코드의 출력 결과:

```bash
python -c 'import torch; print(f"torch: {torch.__version__}")'
python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'
```

* 문제를 재현할 수 있는 Google Colab 노트북 링크

* 불가능할 경우 기존 예제를 사용하여 문제를 재현할 수 있는 표준 및 사용자 지정이 아닌 데이터 집합을 사용할 수 있습니다.

다음 섹션에서는 가장 일반적인 두 가지 문제를 해결하기 위한 가이드를 제공합니다.

### DeepSpeed 프로세스가 시작 단계에서 종료되었을 경우[[deepspeed-process-killed-at-startup]]

실행 중에 트레이스백 없이 DeepSpeed 프로세스가 종료되면 일반적으로 프로그램이 시스템보다 많은 CPU 메모리를 할당하려고 시도했거나 프로세스가 허용된 것보다 많은 CPU 메모리를 할당하려고 시도하여 OS 커널이 프로세스를 종료했음을 의미합니다. 이 경우 구성 파일에 `offload_optimizer`, `offload_param` 또는 둘 다 CPU로 오프로드하도록 구성되어 있는지 확인하세요.  

NVMe 및 ZeRO-3를 설정한 경우 NVMe로 오프로드를 실험해 보세요(모델의 메모리 요구 사항을 [확인](https://deepspeed.readthedocs.io/en/latest/memory.html)하세요).

### NaN 손실[[nan-loss]]

모델을 bf16으로 사전 훈련한 다음 fp16으로 사용하려고 할 때 NaN 손실이 발생하는 경우가 많습니다(특히 TPU 훈련 모델에 해당). 이 문제를 해결하려면 하드웨어가 이를 지원하는 경우(TPU, Ampere GPU 이상) fp32 또는 bf16을 사용하세요.

다른 문제는 fp16 사용과 관련이 있을 수 있습니다. 예를 들어 이것이 fp16 구성인 경우입니다:

```yaml
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

로그에 다음과 같은 `OVERFLOW!` 메시지가 표시될 수 있습니다:

```bash
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

이는 DeepSpeed 손실 스케일러가 손실 오버플로를 극복할 수 있는 스케일링 계수를 찾을 수 없음을 의미합니다. 이 문제를 해결하려면 `initial_scale_power` 값을 더 높게 설정하세요(일반적으로 32가 적절합니다).

## 리소스[[resources]]

DeepSpeed ZeRO는 제한된 GPU 리소스로 추론을 위해 매우 큰 모델을 훈련하고 로드하는 강력한 기술로, 누구나 쉽게 사용할 수 있습니다. DeepSpeed에 대해 자세히 알아보려면 [블로그 포스트](https://www.microsoft.com/en-us/research/search/?q=deepspeed), [공식 문서](https://www.deepspeed.ai/getting-started/), [깃허브 리포지토리](https://github.com/microsoft/deepspeed)를 참조하세요. 

다음 문서도 ZeRO에 대해 자세히 알아볼 수 있는 훌륭한 자료입니다:

* [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://hf.co/papers/1910.02054)
* [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://hf.co/papers/2101.06840)
* [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://hf.co/papers/2104.07857)
