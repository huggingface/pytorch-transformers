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

[పైటోర్చ్](https://pytorch.org/), [టెన్సర్‌ఫ్లో](https://www.tensorflow.org/), మరియు [జాక్స్](https://jax.readthedocs.io/en/latest/) కోసం స్థితి-కలాన యంత్ర అభ్యాసం.

🤗 ట్రాన్స్ఫార్మర్స్ అభివృద్ధిస్తున్నది API మరియు ఉపకరణాలు, పూర్వ-చేతన మోడల్లను సులభంగా డౌన్లోడ్ మరియు శిక్షణ చేయడానికి అవసరమైన సమయం, వనరులు, మరియు వస్తువులను నుంచి మోడల్ను శీర్షికం నుంచి ప్రశిక్షించడం వరకు దేవాయనం చేస్తుంది. ఈ మోడల్లు విభిన్న మోడాలిటీలలో సాధారణ పనులకు మద్దతు చేస్తాయి, వంటివి:

📝 **ప్రాకృతిక భాష ప్రక్రియ**: వచన వర్గీకరణ, పేరుల యొక్క యెంటిటీ గుర్తువు, ప్రశ్న సంవాద, భాషా రచన, సంక్షేపణ, అనువాదం, అనేక ప్రకారాలు, మరియు వచన సృష్టి.<br>
🖼️ **కంప్యూటర్ విషయం**: చిత్రం వర్గీకరణ, వస్త్రం గుర్తువు, మరియు విభజన.<br>
🗣️ **ఆడియో**: స్వయంచలన ప్రసంగాన్ని గుర్తుచేసేందుకు, ఆడియో వర్గీకరణ.<br>
🐙 **బహుమూలిక**: పట్టి ప్రశ్న సంవాద, ఆప్టికల్ సిఫర్ గుర్తువు, డాక్యుమెంట్లు స్క్యాన్ చేసినంతగా సమాచార పొందడం, వీడియో వర్గీకరణ, మరియు దృశ్య ప్రశ్న సంవాద.

🤗 ట్రాన్స్ఫార్మర్స్ పైన మద్దతు చేస్తుంది పైన తొలగించడానికి పైన పైన పైన ప్రోగ్రామ్లో మోడల్ను శిక్షించండి, మరియు అన్ని ప్రాథమిక యొక్కడా ఇన్‌ఫరెన్స్ కోసం లోడ్ చేయండి. మో

డల్లు కూడా ప్రొడక్షన్ వాతావరణాలలో వాడుకోవడానికి ONNX మరియు TorchScript వంటి ఆకృతులకు ఎగుమతి చేయవచ్చు.

ఈరువులకు [హబ్](https://huggingface.co/models), [ఫోరం](https://discuss.huggingface.co/), లేదా [డిస్కార్డ్](https://discord.com/invite/JfAtkvEtRb) లో ఈ పెద్ద సముదాయంలో చేరండి!

## మీరు హగ్గింగ్ ఫేస్ టీమ్ నుండి అనుకూల మద్దతు కోసం చూస్తున్నట్లయితే

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Expert Acceleration Program" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## విషయాలు

డాక్యుమెంటేషన్ ఐదు విభాగాలుగా నిర్వహించబడింది:

- **ప్రారంభించండి** లైబ్రరీ యొక్క శీఘ్ర పర్యటన మరియు రన్నింగ్ కోసం ఇన్‌స్టాలేషన్ సూచనలను అందిస్తుంది.
- **ట్యుటోరియల్స్** మీరు అనుభవశూన్యుడు అయితే ప్రారంభించడానికి గొప్ప ప్రదేశం. మీరు లైబ్రరీని ఉపయోగించడం ప్రారంభించడానికి అవసరమైన ప్రాథమిక నైపుణ్యాలను పొందడానికి ఈ విభాగం మీకు సహాయం చేస్తుంది.
- **హౌ-టు-గైడ్‌లు** లాంగ్వేజ్ మోడలింగ్ కోసం ప్రిట్రైన్డ్ మోడల్‌ని ఫైన్‌ట్యూన్ చేయడం లేదా కస్టమ్ మోడల్‌ను ఎలా వ్రాయాలి మరియు షేర్ చేయాలి వంటి నిర్దిష్ట లక్ష్యాన్ని ఎలా సాధించాలో మీకు చూపుతాయి.
- **కాన్సెప్చువల్ గైడ్స్** మోడల్‌లు, టాస్క్‌లు మరియు 🤗 ట్రాన్స్‌ఫార్మర్ల డిజైన్ ఫిలాసఫీ వెనుక ఉన్న అంతర్లీన భావనలు మరియు ఆలోచనల గురించి మరింత చర్చ మరియు వివరణను అందిస్తుంది.
- **API** అన్ని తరగతులు మరియు విధులను వివరిస్తుంది:

  - **ప్రధాన తరగతులు** కాన్ఫిగరేషన్, మోడల్, టోకెనైజర్ మరియు పైప్‌లైన్ వంటి అత్యంత ముఖ్యమైన తరగతులను వివరిస్తుంది.
  - **మోడల్స్** లైబ్రరీలో అమలు చేయబడిన ప్రతి మోడల్‌కు సంబంధించిన తరగతులు మరియు విధులను వివరిస్తుంది.
  - **అంతర్గత సహాయకులు** అంతర్గతంగా ఉపయోగించే యుటిలిటీ క్లాస్‌లు మరియు ఫంక్షన్‌ల వివరాలు.
 
## మద్దతు ఉన్న నమూనాలు మరియు ఫ్రేమ్‌వర్క్‌లు

దిగువన ఉన్న పట్టిక ఆ ప్రతి మోడల్‌కు పైథాన్ కలిగి ఉన్నా లైబ్రరీలో ప్రస్తుత మద్దతును సూచిస్తుంది
టోకెనైజర్ ("నెమ్మదిగా" అని పిలుస్తారు). Jax (ద్వారా
ఫ్లాక్స్), పైటార్చ్ మరియు/లేదా టెన్సర్‌ఫ్లో.

<!--This table is updated automatically from the auto modules with _make fix-copies_. Do not update manually!-->

|                                  Model                                   | PyTorch support | TensorFlow support | Flax Support |
|:------------------------------------------------------------------------:|:---------------:|:------------------:|:------------:|
|                        [ALBERT](model_doc/albert)                        |       ✅        |         ✅         |      ✅      |
|                         [ALIGN](model_doc/align)                         |       ✅        |         ❌         |      ❌      |
|                       [AltCLIP](model_doc/altclip)                       |       ✅        |         ❌         |      ❌      |
| [Audio Spectrogram Transformer](model_doc/audio-spectrogram-transformer) |       ✅        |         ❌         |      ❌      |
|                    [Autoformer](model_doc/autoformer)                    |       ✅        |         ❌         |      ❌      |
|                          [Bark](model_doc/bark)                          |       ✅        |         ❌         |      ❌      |
|                          [BART](model_doc/bart)                          |       ✅        |         ✅         |      ✅      |
|                       [BARThez](model_doc/barthez)                       |       ✅        |         ✅         |      ✅      |
|                       [BARTpho](model_doc/bartpho)                       |       ✅        |         ✅         |      ✅      |
|                          [BEiT](model_doc/beit)                          |       ✅        |         ❌         |      ✅      |
|                          [BERT](model_doc/bert)                          |       ✅        |         ✅         |      ✅      |
|               [Bert Generation](model_doc/bert-generation)               |       ✅        |         ❌         |      ❌      |
|                 [BertJapanese](model_doc/bert-japanese)                  |       ✅        |         ✅         |      ✅      |
|                      [BERTweet](model_doc/bertweet)                      |       ✅        |         ✅         |      ✅      |
|                      [BigBird](model_doc/big_bird)                       |       ✅        |         ❌         |      ✅      |
|               [BigBird-Pegasus](model_doc/bigbird_pegasus)               |       ✅        |         ❌         |      ❌      |
|                        [BioGpt](model_doc/biogpt)                        |       ✅        |         ❌         |      ❌      |
|                           [BiT](model_doc/bit)                           |       ✅        |         ❌         |      ❌      |
|                    [Blenderbot](model_doc/blenderbot)                    |       ✅        |         ✅         |      ✅      |
|              [BlenderbotSmall](model_doc/blenderbot-small)               |       ✅        |         ✅         |      ✅      |
|                          [BLIP](model_doc/blip)                          |       ✅        |         ✅         |      ❌      |
|                        [BLIP-2](model_doc/blip-2)                        |       ✅        |         ❌         |      ❌      |
|                         [BLOOM](model_doc/bloom)                         |       ✅        |         ❌         |      ✅      |
|                          [BORT](model_doc/bort)                          |       ✅        |         ✅         |      ✅      |
|                   [BridgeTower](model_doc/bridgetower)                   |       ✅        |         ❌         |      ❌      |
|                          [BROS](model_doc/bros)                          |       ✅        |         ❌         |      ❌      |
|                          [ByT5](model_doc/byt5)                          |       ✅        |         ✅         |      ✅      |
|                     [CamemBERT](model_doc/camembert)                     |       ✅        |         ✅         |      ❌      |
|                        [CANINE](model_doc/canine)                        |       ✅        |         ❌         |      ❌      |
|                  [Chinese-CLIP](model_doc/chinese_clip)                  |       ✅        |         ❌         |      ❌      |
|                          [CLAP](model_doc/clap)                          |       ✅        |         ❌         |      ❌      |
|                          [CLIP](model_doc/clip)                          |       ✅        |         ✅         |      ✅      |
|                       [CLIPSeg](model_doc/clipseg)                       |       ✅        |         ❌         |      ❌      |
|                       [CodeGen](model_doc/codegen)                       |       ✅        |         ❌         |      ❌      |
|                    [CodeLlama](model_doc/code_llama)                     |       ✅        |         ❌         |      ❌      |
|              [Conditional DETR](model_doc/conditional_detr)              |       ✅        |         ❌         |      ❌      |
|                      [ConvBERT](model_doc/convbert)                      |       ✅        |         ✅         |      ❌      |
|                      [ConvNeXT](model_doc/convnext)                      |       ✅        |         ✅         |      ❌      |
|                    [ConvNeXTV2](model_doc/convnextv2)                    |       ✅        |         ❌         |      ❌      |
|                           [CPM](model_doc/cpm)                           |       ✅        |         ✅         |      ✅      |
|                       [CPM-Ant](model_doc/cpmant)                        |       ✅        |         ❌         |      ❌      |
|                          [CTRL](model_doc/ctrl)                          |       ✅        |         ✅         |      ❌      |
|                           [CvT](model_doc/cvt)                           |       ✅        |         ✅         |      ❌      |
|                   [Data2VecAudio](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                    [Data2VecText](model_doc/data2vec)                    |       ✅        |         ❌         |      ❌      |
|                   [Data2VecVision](model_doc/data2vec)                   |       ✅        |         ✅         |      ❌      |
|                       [DeBERTa](model_doc/deberta)                       |       ✅        |         ✅         |      ❌      |
|                    [DeBERTa-v2](model_doc/deberta-v2)                    |       ✅        |         ✅         |      ❌      |
|          [Decision Transformer](model_doc/decision_transformer)          |       ✅        |         ❌         |      ❌      |
|               [Deformable DETR](model_doc/deformable_detr)               |       ✅        |         ❌         |      ❌      |
|                          [DeiT](model_doc/deit)                          |       ✅        |         ✅         |      ❌      |
|                        [DePlot](model_doc/deplot)                        |       ✅        |         ❌         |      ❌      |
|                          [DETA](model_doc/deta)                          |       ✅        |         ❌         |      ❌      |
|                          [DETR](model_doc/detr)                          |       ✅        |         ❌         |      ❌      |
|                      [DialoGPT](model_doc/dialogpt)                      |       ✅        |         ✅         |      ✅      |
|                         [DiNAT](model_doc/dinat)                         |       ✅        |         ❌         |      ❌      |
|                        [DINOv2](model_doc/dinov2)                        |       ✅        |         ❌         |      ❌      |
|                    [DistilBERT](model_doc/distilbert)                    |       ✅        |         ✅         |      ✅      |
|                           [DiT](model_doc/dit)                           |       ✅        |         ❌         |      ✅      |
|                       [DonutSwin](model_doc/donut)                       |       ✅        |         ❌         |      ❌      |
|                           [DPR](model_doc/dpr)                           |       ✅        |         ✅         |      ❌      |
|                           [DPT](model_doc/dpt)                           |       ✅        |         ❌         |      ❌      |
|               [EfficientFormer](model_doc/efficientformer)               |       ✅        |         ✅         |      ❌      |
|                  [EfficientNet](model_doc/efficientnet)                  |       ✅        |         ❌         |      ❌      |
|                       [ELECTRA](model_doc/electra)                       |       ✅        |         ✅         |      ✅      |
|                       [EnCodec](model_doc/encodec)                       |       ✅        |         ❌         |      ❌      |
|               [Encoder decoder](model_doc/encoder-decoder)               |       ✅        |         ✅         |      ✅      |
|                         [ERNIE](model_doc/ernie)                         |       ✅        |         ❌         |      ❌      |
|                       [ErnieM](model_doc/ernie_m)                        |       ✅        |         ❌         |      ❌      |
|                           [ESM](model_doc/esm)                           |       ✅        |         ✅         |      ❌      |
|              [FairSeq Machine-Translation](model_doc/fsmt)               |       ✅        |         ❌         |      ❌      |
|                        [Falcon](model_doc/falcon)                        |       ✅        |         ❌         |      ❌      |
|                       [FLAN-T5](model_doc/flan-t5)                       |       ✅        |         ✅         |      ✅      |
|                      [FLAN-UL2](model_doc/flan-ul2)                      |       ✅        |         ✅         |      ✅      |
|                      [FlauBERT](model_doc/flaubert)                      |       ✅        |         ✅         |      ❌      |
|                         [FLAVA](model_doc/flava)                         |       ✅        |         ❌         |      ❌      |
|                          [FNet](model_doc/fnet)                          |       ✅        |         ❌         |      ❌      |
|                      [FocalNet](model_doc/focalnet)                      |       ✅        |         ❌         |      ❌      |
|                  [Funnel Transformer](model_doc/funnel)                  |       ✅        |         ✅         |      ❌      |
|                           [GIT](model_doc/git)                           |       ✅        |         ❌         |      ❌      |
|                          [GLPN](model_doc/glpn)                          |       ✅        |         ❌         |      ❌      |
|                       [GPT Neo](model_doc/gpt_neo)                       |       ✅        |         ❌         |      ✅      |
|                      [GPT NeoX](model_doc/gpt_neox)                      |       ✅        |         ❌         |      ❌      |
|             [GPT NeoX Japanese](model_doc/gpt_neox_japanese)             |       ✅        |         ❌         |      ❌      |
|                         [GPT-J](model_doc/gptj)                          |       ✅        |         ✅         |      ✅      |
|                       [GPT-Sw3](model_doc/gpt-sw3)                       |       ✅        |         ✅         |      ✅      |
|                   [GPTBigCode](model_doc/gpt_bigcode)                    |       ✅        |         ❌         |      ❌      |
|               [GPTSAN-japanese](model_doc/gptsan-japanese)               |       ✅        |         ❌         |      ❌      |
|                    [Graphormer](model_doc/graphormer)                    |       ✅        |         ❌         |      ❌      |
|                      [GroupViT](model_doc/groupvit)                      |       ✅        |         ✅         |      ❌      |
|                       [HerBERT](model_doc/herbert)                       |       ✅        |         ✅         |      ✅      |
|                        [Hubert](model_doc/hubert)                        |       ✅        |         ✅         |      ❌      |
|                        [I-BERT](model_doc/ibert)                         |       ✅        |         ❌         |      ❌      |
|                       [IDEFICS](model_doc/idefics)                       |       ✅        |         ❌         |      ❌      |
|                      [ImageGPT](model_doc/imagegpt)                      |       ✅        |         ❌         |      ❌      |
|                      [Informer](model_doc/informer)                      |       ✅        |         ❌         |      ❌      |
|                  [InstructBLIP](model_doc/instructblip)                  |       ✅        |         ❌         |      ❌      |
|                       [Jukebox](model_doc/jukebox)                       |       ✅        |         ❌         |      ❌      |
|                      [LayoutLM](model_doc/layoutlm)                      |       ✅        |         ✅         |      ❌      |
|                    [LayoutLMv2](model_doc/layoutlmv2)                    |       ✅        |         ❌         |      ❌      |
|                    [LayoutLMv3](model_doc/layoutlmv3)                    |       ✅        |         ✅         |      ❌      |
|                     [LayoutXLM](model_doc/layoutxlm)                     |       ✅        |         ❌         |      ❌      |
|                           [LED](model_doc/led)                           |       ✅        |         ✅         |      ❌      |
|                         [LeViT](model_doc/levit)                         |       ✅        |         ❌         |      ❌      |
|                          [LiLT](model_doc/lilt)                          |       ✅        |         ❌         |      ❌      |
|                         [LLaMA](model_doc/llama)                         |       ✅        |         ❌         |      ❌      |
|                        [Llama2](model_doc/llama2)                        |       ✅        |         ❌         |      ❌      |
|                    [Longformer](model_doc/longformer)                    |       ✅        |         ✅         |      ❌      |
|                        [LongT5](model_doc/longt5)                        |       ✅        |         ❌         |      ✅      |
|                          [LUKE](model_doc/luke)                          |       ✅        |         ❌         |      ❌      |
|                        [LXMERT](model_doc/lxmert)                        |       ✅        |         ✅         |      ❌      |
|                        [M-CTC-T](model_doc/mctct)                        |       ✅        |         ❌         |      ❌      |
|                       [M2M100](model_doc/m2m_100)                        |       ✅        |         ❌         |      ❌      |
|                        [Marian](model_doc/marian)                        |       ✅        |         ✅         |      ✅      |
|                      [MarkupLM](model_doc/markuplm)                      |       ✅        |         ❌         |      ❌      |
|                   [Mask2Former](model_doc/mask2former)                   |       ✅        |         ❌         |      ❌      |
|                    [MaskFormer](model_doc/maskformer)                    |       ✅        |         ❌         |      ❌      |
|                        [MatCha](model_doc/matcha)                        |       ✅        |         ❌         |      ❌      |
|                         [mBART](model_doc/mbart)                         |       ✅        |         ✅         |      ✅      |
|                      [mBART-50](model_doc/mbart50)                       |       ✅        |         ✅         |      ✅      |
|                          [MEGA](model_doc/mega)                          |       ✅        |         ❌         |      ❌      |
|                 [Megatron-BERT](model_doc/megatron-bert)                 |       ✅        |         ❌         |      ❌      |
|                 [Megatron-GPT2](model_doc/megatron_gpt2)                 |       ✅        |         ✅         |      ✅      |
|                       [MGP-STR](model_doc/mgp-str)                       |       ✅        |         ❌         |      ❌      |
|                       [Mistral](model_doc/mistral)                       |       ✅        |         ❌         |      ❌      |
|                         [mLUKE](model_doc/mluke)                         |       ✅        |         ❌         |      ❌      |
|                           [MMS](model_doc/mms)                           |       ✅        |         ✅         |      ✅      |
|                    [MobileBERT](model_doc/mobilebert)                    |       ✅        |         ✅         |      ❌      |
|                  [MobileNetV1](model_doc/mobilenet_v1)                   |       ✅        |         ❌         |      ❌      |
|                  [MobileNetV2](model_doc/mobilenet_v2)                   |       ✅        |         ❌         |      ❌      |
|                     [MobileViT](model_doc/mobilevit)                     |       ✅        |         ✅         |      ❌      |
|                   [MobileViTV2](model_doc/mobilevitv2)                   |       ✅        |         ❌         |      ❌      |
|                         [MPNet](model_doc/mpnet)                         |       ✅        |         ✅         |      ❌      |
|                           [MPT](model_doc/mpt)                           |       ✅        |         ❌         |      ❌      |
|                           [MRA](model_doc/mra)                           |       ✅        |         ❌         |      ❌      |
|                           [MT5](model_doc/mt5)                           |       ✅        |         ✅         |      ✅      |
|                      [MusicGen](model_doc/musicgen)                      |       ✅        |         ❌         |      ❌      |
|                           [MVP](model_doc/mvp)                           |       ✅        |         ❌         |      ❌      |
|                           [NAT](model_doc/nat)                           |       ✅        |         ❌         |      ❌      |
|                         [Nezha](model_doc/nezha)                         |       ✅        |         ❌         |      ❌      |
|                          [NLLB](model_doc/nllb)                          |       ✅        |         ❌         |      ❌      |
|                      [NLLB-MOE](model_doc/nllb-moe)                      |       ✅        |         ❌         |      ❌      |
|                        [Nougat](model_doc/nougat)                        |       ✅        |         ✅         |      ✅      |
|                 [Nyströmformer](model_doc/nystromformer)                 |       ✅        |         ❌         |      ❌      |
|                     [OneFormer](model_doc/oneformer)                     |       ✅        |         ❌         |      ❌      |
|                    [OpenAI GPT](model_doc/openai-gpt)                    |       ✅        |         ✅         |      ❌      |
|                      [OpenAI GPT-2](model_doc/gpt2)                      |       ✅        |         ✅         |      ✅      |
|                    [OpenLlama](model_doc/open-llama)                     |       ✅        |         ❌         |      ❌      |
|                           [OPT](model_doc/opt)                           |       ✅        |         ✅         |      ✅      |
|                       [OWL-ViT](model_doc/owlvit)                        |       ✅        |         ❌         |      ❌      |
|                       [Pegasus](model_doc/pegasus)                       |       ✅        |         ✅         |      ✅      |
|                     [PEGASUS-X](model_doc/pegasus_x)                     |       ✅        |         ❌         |      ❌      |
|                     [Perceiver](model_doc/perceiver)                     |       ✅        |         ❌         |      ❌      |
|                     [Persimmon](model_doc/persimmon)                     |       ✅        |         ❌         |      ❌      |
|                       [PhoBERT](model_doc/phobert)                       |       ✅        |         ✅         |      ✅      |
|                    [Pix2Struct](model_doc/pix2struct)                    |       ✅        |         ❌         |      ❌      |
|                        [PLBart](model_doc/plbart)                        |       ✅        |         ❌         |      ❌      |
|                    [PoolFormer](model_doc/poolformer)                    |       ✅        |         ❌         |      ❌      |
|                     [Pop2Piano](model_doc/pop2piano)                     |       ✅        |         ❌         |      ❌      |
|                    [ProphetNet](model_doc/prophetnet)                    |       ✅        |         ❌         |      ❌      |
|                           [PVT](model_doc/pvt)                           |       ✅        |         ❌         |      ❌      |
|                       [QDQBert](model_doc/qdqbert)                       |       ✅        |         ❌         |      ❌      |
|                           [RAG](model_doc/rag)                           |       ✅        |         ✅         |      ❌      |
|                         [REALM](model_doc/realm)                         |       ✅        |         ❌         |      ❌      |
|                      [Reformer](model_doc/reformer)                      |       ✅        |         ❌         |      ❌      |
|                        [RegNet](model_doc/regnet)                        |       ✅        |         ✅         |      ✅      |
|                       [RemBERT](model_doc/rembert)                       |       ✅        |         ✅         |      ❌      |
|                        [ResNet](model_doc/resnet)                        |       ✅        |         ✅         |      ✅      |
|                     [RetriBERT](model_doc/retribert)                     |       ✅        |         ❌         |      ❌      |
|                       [RoBERTa](model_doc/roberta)                       |       ✅        |         ✅         |      ✅      |
|          [RoBERTa-PreLayerNorm](model_doc/roberta-prelayernorm)          |       ✅        |         ✅         |      ✅      |
|                      [RoCBert](model_doc/roc_bert)                       |       ✅        |         ❌         |      ❌      |
|                      [RoFormer](model_doc/roformer)                      |       ✅        |         ✅         |      ✅      |
|                          [RWKV](model_doc/rwkv)                          |       ✅        |         ❌         |      ❌      |
|                         [RWKV6](model_doc/rwkv6)                         |       ✅        |         ❌         |      ❌      |
|                           [SAM](model_doc/sam)                           |       ✅        |         ✅         |      ❌      |
|                     [SegFormer](model_doc/segformer)                     |       ✅        |         ✅         |      ❌      |
|                           [SEW](model_doc/sew)                           |       ✅        |         ❌         |      ❌      |
|                         [SEW-D](model_doc/sew-d)                         |       ✅        |         ❌         |      ❌      |
|        [Speech Encoder decoder](model_doc/speech-encoder-decoder)        |       ✅        |         ❌         |      ✅      |
|                 [Speech2Text](model_doc/speech_to_text)                  |       ✅        |         ✅         |      ❌      |
|                      [SpeechT5](model_doc/speecht5)                      |       ✅        |         ❌         |      ❌      |
|                      [Splinter](model_doc/splinter)                      |       ✅        |         ❌         |      ❌      |
|                   [SqueezeBERT](model_doc/squeezebert)                   |       ✅        |         ❌         |      ❌      |
|                   [SwiftFormer](model_doc/swiftformer)                   |       ✅        |         ❌         |      ❌      |
|                    [Swin Transformer](model_doc/swin)                    |       ✅        |         ✅         |      ❌      |
|                 [Swin Transformer V2](model_doc/swinv2)                  |       ✅        |         ❌         |      ❌      |
|                       [Swin2SR](model_doc/swin2sr)                       |       ✅        |         ❌         |      ❌      |
|           [SwitchTransformers](model_doc/switch_transformers)            |       ✅        |         ❌         |      ❌      |
|                            [T5](model_doc/t5)                            |       ✅        |         ✅         |      ✅      |
|                        [T5v1.1](model_doc/t5v1.1)                        |       ✅        |         ✅         |      ✅      |
|             [Table Transformer](model_doc/table-transformer)             |       ✅        |         ❌         |      ❌      |
|                         [TAPAS](model_doc/tapas)                         |       ✅        |         ✅         |      ❌      |
|                         [TAPEX](model_doc/tapex)                         |       ✅        |         ✅         |      ✅      |
|       [Time Series Transformer](model_doc/time_series_transformer)       |       ✅        |         ❌         |      ❌      |
|                   [TimeSformer](model_doc/timesformer)                   |       ✅        |         ❌         |      ❌      |
|        [Trajectory Transformer](model_doc/trajectory_transformer)        |       ✅        |         ❌         |      ❌      |
|                  [Transformer-XL](model_doc/transfo-xl)                  |       ✅        |         ✅         |      ❌      |
|                         [TrOCR](model_doc/trocr)                         |       ✅        |         ❌         |      ❌      |
|                          [TVLT](model_doc/tvlt)                          |       ✅        |         ❌         |      ❌      |
|                           [UL2](model_doc/ul2)                           |       ✅        |         ✅         |      ✅      |
|                          [UMT5](model_doc/umt5)                          |       ✅        |         ❌         |      ❌      |
|                     [UniSpeech](model_doc/unispeech)                     |       ✅        |         ❌         |      ❌      |
|                 [UniSpeechSat](model_doc/unispeech-sat)                  |       ✅        |         ❌         |      ❌      |
|                       [UPerNet](model_doc/upernet)                       |       ✅        |         ❌         |      ❌      |
|                           [VAN](model_doc/van)                           |       ✅        |         ❌         |      ❌      |
|                      [VideoMAE](model_doc/videomae)                      |       ✅        |         ❌         |      ❌      |
|                          [ViLT](model_doc/vilt)                          |       ✅        |         ❌         |      ❌      |
|        [Vision Encoder decoder](model_doc/vision-encoder-decoder)        |       ✅        |         ✅         |      ✅      |
|       [VisionTextDualEncoder](model_doc/vision-text-dual-encoder)        |       ✅        |         ✅         |      ✅      |
|                   [VisualBERT](model_doc/visual_bert)                    |       ✅        |         ❌         |      ❌      |
|                           [ViT](model_doc/vit)                           |       ✅        |         ✅         |      ✅      |
|                    [ViT Hybrid](model_doc/vit_hybrid)                    |       ✅        |         ❌         |      ❌      |
|                        [VitDet](model_doc/vitdet)                        |       ✅        |         ❌         |      ❌      |
|                       [ViTMAE](model_doc/vit_mae)                        |       ✅        |         ✅         |      ❌      |
|                      [ViTMatte](model_doc/vitmatte)                      |       ✅        |         ❌         |      ❌      |
|                       [ViTMSN](model_doc/vit_msn)                        |       ✅        |         ❌         |      ❌      |
|                          [VITS](model_doc/vits)                          |       ✅        |         ❌         |      ❌      |
|                         [ViViT](model_doc/vivit)                         |       ✅        |         ❌         |      ❌      |
|                      [Wav2Vec2](model_doc/wav2vec2)                      |       ✅        |         ✅         |      ✅      |
|            [Wav2Vec2-Conformer](model_doc/wav2vec2-conformer)            |       ✅        |         ❌         |      ❌      |
|              [Wav2Vec2Phoneme](model_doc/wav2vec2_phoneme)               |       ✅        |         ✅         |      ✅      |
|                         [WavLM](model_doc/wavlm)                         |       ✅        |         ❌         |      ❌      |
|                       [Whisper](model_doc/whisper)                       |       ✅        |         ✅         |      ✅      |
|                        [X-CLIP](model_doc/xclip)                         |       ✅        |         ❌         |      ❌      |
|                         [X-MOD](model_doc/xmod)                          |       ✅        |         ❌         |      ❌      |
|                          [XGLM](model_doc/xglm)                          |       ✅        |         ✅         |      ✅      |
|                           [XLM](model_doc/xlm)                           |       ✅        |         ✅         |      ❌      |
|                [XLM-ProphetNet](model_doc/xlm-prophetnet)                |       ✅        |         ❌         |      ❌      |
|                   [XLM-RoBERTa](model_doc/xlm-roberta)                   |       ✅        |         ✅         |      ✅      |
|                [XLM-RoBERTa-XL](model_doc/xlm-roberta-xl)                |       ✅        |         ❌         |      ❌      |
|                         [XLM-V](model_doc/xlm-v)                         |       ✅        |         ✅         |      ✅      |
|                         [XLNet](model_doc/xlnet)                         |       ✅        |         ✅         |      ❌      |
|                         [XLS-R](model_doc/xls_r)                         |       ✅        |         ✅         |      ✅      |
|                 [XLSR-Wav2Vec2](model_doc/xlsr_wav2vec2)                 |       ✅        |         ✅         |      ✅      |
|                         [YOLOS](model_doc/yolos)                         |       ✅        |         ❌         |      ❌      |
|                          [YOSO](model_doc/yoso)                          |       ✅        |         ❌         |      ❌      |

<!-- End table-->
