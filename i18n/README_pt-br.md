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
-->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <a href="https://github.com/huggingface/transformers/">English</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <b>Рortuguês</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

<h3 align="center">
    <p>Aprendizado de máquina de última geração para JAX, PyTorch e TensorFlow</p>
</h3>

<h3 align="center">
    <a href="https://hf.co/course"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/course_banner.png"></a>
</h3>


A biblioteca 🤗 Transformers oferece milhares de modelos pré-treinados para executar tarefas em diferentes modalidades, como texto, visão e áudio.

Esses modelos podem ser aplicados a:

* 📝 Texto, para tarefas como classificação de texto, extração de informações, resposta a perguntas, sumarização, tradução, geração de texto, em mais de 100 idiomas.
* 🖼️ Imagens, para tarefas como classificação de imagens, detecção de objetos e segmentação.
* 🗣️ Áudio, para tarefas como reconhecimento de fala e classificação de áudio.

Os modelos Transformer também podem executar tarefas em diversas modalidades combinadas, como responder a perguntas em tabelas, reconhecimento óptico de caracteres, extração de informações de documentos digitalizados, classificação de vídeo e resposta a perguntas visuais.


A biblioteca 🤗 Transformers oferece APIs para baixar e usar rapidamente esses modelos pré-treinados em um texto específico, ajustá-los em seus próprios conjuntos de dados e, em seguida, compartilhá-los com a comunidade em nosso [model hub](https://huggingface.co/models). Ao mesmo tempo, cada módulo Python que define uma arquitetura é totalmente independente e pode ser modificado para permitir experimentos de pesquisa rápidos.

A biblioteca 🤗 Transformers é respaldada pelas três bibliotecas de aprendizado profundo mais populares — [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) e [TensorFlow](https://www.tensorflow.org/) — com uma integração perfeita entre elas. É simples treinar seus modelos com uma delas antes de carregá-los para inferência com a outra

## Demonstração Online

Você pode testar a maioria de nossos modelos diretamente em suas páginas a partir do [model hub](https://huggingface.co/models). Também oferecemos [hospedagem de modelos privados, versionamento e uma API de inferência](https://huggingface.co/pricing)
para modelos públicos e privados.

Aqui estão alguns exemplos:

Em Processamento de Linguagem Natural:

- [Completar palavra mascarada com BERT](https://huggingface.co/google-bert/bert-base-uncased?text=Paris+is+the+%5BMASK%5D+of+France)
- [Reconhecimento de Entidades Nomeadas com Electra](https://huggingface.co/dbmdz/electra-large-discriminator-finetuned-conll03-english?text=My+name+is+Sarah+and+I+live+in+London+city)
- [Geração de texto com GPT-2](https://huggingface.co/openai-community/gpt2?text=A+long+time+ago%2C)
- [Inferência de Linguagem Natural com RoBERTa](https://huggingface.co/FacebookAI/roberta-large-mnli?text=The+dog+was+lost.+Nobody+lost+any+animal)
- [Sumarização com BART](https://huggingface.co/facebook/bart-large-cnn?text=The+tower+is+324+metres+%281%2C063+ft%29+tall%2C+about+the+same+height+as+an+81-storey+building%2C+and+the+tallest+structure+in+Paris.+Its+base+is+square%2C+measuring+125+metres+%28410+ft%29+on+each+side.+During+its+construction%2C+the+Eiffel+Tower+surpassed+the+Washington+Monument+to+become+the+tallest+man-made+structure+in+the+world%2C+a+title+it+held+for+41+years+until+the+Chrysler+Building+in+New+York+City+was+finished+in+1930.+It+was+the+first+structure+to+reach+a+height+of+300+metres.+Due+to+the+addition+of+a+broadcasting+aerial+at+the+top+of+the+tower+in+1957%2C+it+is+now+taller+than+the+Chrysler+Building+by+5.2+metres+%2817+ft%29.+Excluding+transmitters%2C+the+Eiffel+Tower+is+the+second+tallest+free-standing+structure+in+France+after+the+Millau+Viaduct)
- [Resposta a perguntas com DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad?text=Which+name+is+also+used+to+describe+the+Amazon+rainforest+in+English%3F&context=The+Amazon+rainforest+%28Portuguese%3A+Floresta+Amaz%C3%B4nica+or+Amaz%C3%B4nia%3B+Spanish%3A+Selva+Amaz%C3%B3nica%2C+Amazon%C3%ADa+or+usually+Amazonia%3B+French%3A+For%C3%AAt+amazonienne%3B+Dutch%3A+Amazoneregenwoud%29%2C+also+known+in+English+as+Amazonia+or+the+Amazon+Jungle%2C+is+a+moist+broadleaf+forest+that+covers+most+of+the+Amazon+basin+of+South+America.+This+basin+encompasses+7%2C000%2C000+square+kilometres+%282%2C700%2C000+sq+mi%29%2C+of+which+5%2C500%2C000+square+kilometres+%282%2C100%2C000+sq+mi%29+are+covered+by+the+rainforest.+This+region+includes+territory+belonging+to+nine+nations.+The+majority+of+the+forest+is+contained+within+Brazil%2C+with+60%25+of+the+rainforest%2C+followed+by+Peru+with+13%25%2C+Colombia+with+10%25%2C+and+with+minor+amounts+in+Venezuela%2C+Ecuador%2C+Bolivia%2C+Guyana%2C+Suriname+and+French+Guiana.+States+or+departments+in+four+nations+contain+%22Amazonas%22+in+their+names.+The+Amazon+represents+over+half+of+the+planet%27s+remaining+rainforests%2C+and+comprises+the+largest+and+most+biodiverse+tract+of+tropical+rainforest+in+the+world%2C+with+an+estimated+390+billion+individual+trees+divided+into+16%2C000+species)
- [Tradução com T5](https://huggingface.co/google-t5/t5-base?text=My+name+is+Wolfgang+and+I+live+in+Berlin)


Em Visão Computacional:
- [Classificação de Imagens com ViT](https://huggingface.co/google/vit-base-patch16-224)
- [Detecção de Objetos com DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Segmentação Semântica com SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- [Segmentação Panóptica com MaskFormer](https://huggingface.co/facebook/maskformer-swin-small-coco)
- [Estimativa de Profundidade com DPT](https://huggingface.co/docs/transformers/model_doc/dpt)
- [Classificação de Vídeo com VideoMAE](https://huggingface.co/docs/transformers/model_doc/videomae)
- [Segmentação Universal com OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_dinat_large)


Em Áudio:
- [Reconhecimento Automático de Fala com Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- [Detecção de Palavras-Chave com Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- [Classificação de Áudio com Transformer de Espectrograma de Áudio](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)

Em Tarefas Multimodais:
- [Respostas de Perguntas em Tabelas com TAPAS](https://huggingface.co/google/tapas-base-finetuned-wtq)
- [Respostas de Perguntas Visuais com ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa)
- [Classificação de Imagens sem Anotação com CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
- [Respostas de Perguntas em Documentos com LayoutLM](https://huggingface.co/impira/layoutlm-document-qa)
- [Classificação de Vídeo sem Anotação com X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip)

## 100 Projetos Usando Transformers

Transformers é mais do que um conjunto de ferramentas para usar modelos pré-treinados: é uma comunidade de projetos construídos ao seu redor e o Hugging Face Hub. Queremos que o Transformers permita que desenvolvedores, pesquisadores, estudantes, professores, engenheiros e qualquer outra pessoa construa seus projetos dos sonhos.

Para celebrar as 100.000 estrelas do Transformers, decidimos destacar a comunidade e criamos a página [awesome-transformers](./awesome-transformers.md), que lista 100 projetos incríveis construídos nas proximidades dos Transformers.

Se você possui ou utiliza um projeto que acredita que deveria fazer parte da lista, abra um PR para adicioná-lo!

## Se você está procurando suporte personalizado da equipe Hugging Face

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a><br>


## Tour Rápido

Para usar imediatamente um modelo em uma entrada específica (texto, imagem, áudio, ...), oferecemos a API `pipeline`. Os pipelines agrupam um modelo pré-treinado com o pré-processamento que foi usado durante o treinamento desse modelo. Aqui está como usar rapidamente um pipeline para classificar textos como positivos ou negativos:

```python
from transformers import pipeline

# Carregue o pipeline de classificação de texto
>>> classifier = pipeline("sentiment-analysis")

# Classifique o texto como positivo ou negativo
>>> classifier("Estamos muito felizes em apresentar o pipeline no repositório dos transformers.")
[{'label': 'POSITIVE', 'score': 0.9996980428695679}]
```

A segunda linha de código baixa e armazena em cache o modelo pré-treinado usado pelo pipeline, enquanto a terceira linha o avalia no texto fornecido. Neste exemplo, a resposta é "positiva" com uma confiança de 99,97%.

Muitas tarefas têm um `pipeline` pré-treinado pronto para uso, não apenas em PNL, mas também em visão computacional e processamento de áudio. Por exemplo, podemos facilmente extrair objetos detectados em uma imagem:

``` python
>>> import requests
>>> from PIL import Image
>>> from transformers import pipeline

# Download an image with cute cats
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
>>> image_data = requests.get(url, stream=True).raw
>>> image = Image.open(image_data)

# Allocate a pipeline for object detection
>>> object_detector = pipeline('object-detection')
>>> object_detector(image)
[{'score': 0.9982201457023621,
  'label': 'remote',
  'box': {'xmin': 40, 'ymin': 70, 'xmax': 175, 'ymax': 117}},
 {'score': 0.9960021376609802,
  'label': 'remote',
  'box': {'xmin': 333, 'ymin': 72, 'xmax': 368, 'ymax': 187}},
 {'score': 0.9954745173454285,
  'label': 'couch',
  'box': {'xmin': 0, 'ymin': 1, 'xmax': 639, 'ymax': 473}},
 {'score': 0.9988006353378296,
  'label': 'cat',
  'box': {'xmin': 13, 'ymin': 52, 'xmax': 314, 'ymax': 470}},
 {'score': 0.9986783862113953,
  'label': 'cat',
  'box': {'xmin': 345, 'ymin': 23, 'xmax': 640, 'ymax': 368}}]
```


Aqui obtemos uma lista de objetos detectados na imagem, com uma caixa envolvendo o objeto e uma pontuação de confiança. Aqui está a imagem original à esquerda, com as previsões exibidas à direita:

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png" width="400"></a>
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample_post_processed.png" width="400"></a>
</h3>

Você pode aprender mais sobre as tarefas suportadas pela API `pipeline` em [este tutorial](https://huggingface.co/docs/transformers/task_summary).


Além do `pipeline`, para baixar e usar qualquer um dos modelos pré-treinados em sua tarefa específica, tudo o que é necessário são três linhas de código. Aqui está a versão em PyTorch:

```python
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```

E aqui está o código equivalente para TensorFlow:

```python
>>> from transformers import AutoTokenizer, TFAutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("google-bert/bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

O tokenizador é responsável por todo o pré-processamento que o modelo pré-treinado espera, e pode ser chamado diretamente em uma única string (como nos exemplos acima) ou em uma lista. Ele produzirá um dicionário que você pode usar no código subsequente ou simplesmente passar diretamente para o seu modelo usando o operador de descompactação de argumentos **.

O modelo em si é um [Pytorch `nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)  ou um [TensorFlow `tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)(dependendo do seu back-end) que você pode usar como de costume. [Este tutorial](https://huggingface.co/docs/transformers/training) explica como integrar esse modelo em um ciclo de treinamento clássico do PyTorch ou TensorFlow, ou como usar nossa API `Trainer` para ajuste fino rápido em um novo conjunto de dados.

## Por que devo usar transformers?

1. Modelos state-of-the-art fáceis de usar:
    - Alto desempenho em compreensão e geração de linguagem natural, visão computacional e tarefas de áudio.
    - Barreira de entrada baixa para educadores e profissionais.
    - Poucas abstrações visíveis para o usuário, com apenas três classes para aprender.
    - Uma API unificada para usar todos os nossos modelos pré-treinados.

1. Menores custos de computação, menor pegada de carbono:
    - Pesquisadores podem compartilhar modelos treinados em vez de treinar sempre do zero.
    - Profissionais podem reduzir o tempo de computação e os custos de produção.
    - Dezenas de arquiteturas com mais de 60.000 modelos pré-treinados em todas as modalidades.

1. Escolha o framework certo para cada parte da vida de um modelo:
    - Treine modelos state-of-the-art em 3 linhas de código.
    - Mova um único modelo entre frameworks TF2.0/PyTorch/JAX à vontade.
    - Escolha o framework certo de forma contínua para treinamento, avaliação e produção.

1. Personalize facilmente um modelo ou um exemplo para atender às suas necessidades:
    - Fornecemos exemplos para cada arquitetura para reproduzir os resultados publicados pelos autores originais.
    - Os detalhes internos do modelo são expostos de maneira consistente.
    - Os arquivos do modelo podem ser usados de forma independente da biblioteca para experimentos rápidos.

## Por que não devo usar transformers?

- Esta biblioteca não é uma caixa de ferramentas modular para construir redes neurais. O código nos arquivos do modelo não é refatorado com abstrações adicionais de propósito, para que os pesquisadores possam iterar rapidamente em cada um dos modelos sem se aprofundar em abstrações/arquivos adicionais.
- A API de treinamento não é projetada para funcionar com qualquer modelo, mas é otimizada para funcionar com os modelos fornecidos pela biblioteca. Para loops de aprendizado de máquina genéricos, você deve usar outra biblioteca (possivelmente, [Accelerate](https://huggingface.co/docs/accelerate)).
- Embora nos esforcemos para apresentar o maior número possível de casos de uso, os scripts em nossa [pasta de exemplos](https://github.com/huggingface/transformers/tree/main/examples) são apenas isso: exemplos. É esperado que eles não funcionem prontos para uso em seu problema específico e que seja necessário modificar algumas linhas de código para adaptá-los às suas necessidades.



### Com pip

Este repositório é testado no Python 3.9+, Flax 0.4.1+, PyTorch 2.0+ e TensorFlow 2.6+.

Você deve instalar o 🤗 Transformers em um [ambiente virtual](https://docs.python.org/3/library/venv.html). Se você não está familiarizado com ambientes virtuais em Python, confira o [guia do usuário](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Primeiro, crie um ambiente virtual com a versão do Python que você vai usar e ative-o.

Em seguida, você precisará instalar pelo menos um dos back-ends Flax, PyTorch ou TensorFlow.
Consulte a [página de instalação do TensorFlow](https://www.tensorflow.org/install/), a [página de instalação do PyTorch](https://pytorch.org/get-started/locally/#start-locally) e/ou [Flax](https://github.com/google/flax#quick-install) e [Jax](https://github.com/google/jax#installation) páginas de instalação para obter o comando de instalação específico para a sua plataforma.

Quando um desses back-ends estiver instalado, o 🤗 Transformers pode ser instalado usando pip da seguinte forma:

```bash
pip install transformers
```
Se você deseja experimentar com os exemplos ou precisa da versão mais recente do código e não pode esperar por um novo lançamento, você deve instalar a [biblioteca a partir do código-fonte](https://huggingface.co/docs/transformers/installation#installing-from-source).

### Com conda

O 🤗 Transformers pode ser instalado com conda da seguinte forma:

```bash
conda install conda-forge::transformers
```

> **_NOTA:_**  Instalar `transformers` pelo canal `huggingface` está obsoleto.

Siga as páginas de instalação do Flax, PyTorch ou TensorFlow para ver como instalá-los com conda.

Siga as páginas de instalação do Flax, PyTorch ou TensorFlow para ver como instalá-los com o conda.

> **_NOTA:_**  No Windows, você pode ser solicitado a ativar o Modo de Desenvolvedor para aproveitar o cache. Se isso não for uma opção para você, por favor nos avise [neste problema](https://github.com/huggingface/huggingface_hub/issues/1062).

## Arquiteturas de Modelos

**[Todos os pontos de verificação de modelo](https://huggingface.co/models)** fornecidos pelo 🤗 Transformers são integrados de forma transparente do [model hub](https://huggingface.co/models) do huggingface.co, onde são carregados diretamente por [usuários](https://huggingface.co/users) e [organizações](https://huggingface.co/organizations).

Número atual de pontos de verificação: ![](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)

🤗 Transformers atualmente fornece as seguintes arquiteturas: veja [aqui](https://huggingface.co/docs/transformers/model_summary) para um resumo de alto nível de cada uma delas.

Para verificar se cada modelo tem uma implementação em Flax, PyTorch ou TensorFlow, ou possui um tokenizador associado com a biblioteca 🤗 Tokenizers, consulte [esta tabela](https://huggingface.co/docs/transformers/index#supported-frameworks).

Essas implementações foram testadas em vários conjuntos de dados (veja os scripts de exemplo) e devem corresponder ao desempenho das implementações originais. Você pode encontrar mais detalhes sobre o desempenho na seção de Exemplos da [documentação](https://github.com/huggingface/transformers/tree/main/examples).


## Saiba mais

| Seção | Descrição |
|-|-|
| [Documentação](https://huggingface.co/docs/transformers/) | Documentação completa da API e tutoriais |
| [Resumo de Tarefas](https://huggingface.co/docs/transformers/task_summary) | Tarefas suportadas pelo 🤗 Transformers |
| [Tutorial de Pré-processamento](https://huggingface.co/docs/transformers/preprocessing) | Usando a classe `Tokenizer` para preparar dados para os modelos |
| [Treinamento e Ajuste Fino](https://huggingface.co/docs/transformers/training) | Usando os modelos fornecidos pelo 🤗 Transformers em um loop de treinamento PyTorch/TensorFlow e a API `Trainer` |
| [Tour Rápido: Scripts de Ajuste Fino/Utilização](https://github.com/huggingface/transformers/tree/main/examples) | Scripts de exemplo para ajuste fino de modelos em uma ampla gama de tarefas |
| [Compartilhamento e Envio de Modelos](https://huggingface.co/docs/transformers/model_sharing) | Envie e compartilhe seus modelos ajustados com a comunidade |

## Citação

Agora temos um [artigo](https://www.aclweb.org/anthology/2020.emnlp-demos.6/) que você pode citar para a biblioteca 🤗 Transformers:
```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = out,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
