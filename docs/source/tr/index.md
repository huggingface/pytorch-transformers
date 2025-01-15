<!--Telif Hakkı 2020 The HuggingFace Ekibi. Tüm hakları saklıdır.

Apache Lisansı, Sürüm 2.0 (Lisans); bu dosyayı yürürlükteki yasalara uygun bir şekilde kullanabilirsiniz. Lisansın bir kopyasını aşağıdaki adresten alabilirsiniz.

http://www.apache.org/licenses/LICENSE-2.0

Lisansa tabi olmayan durumlarda veya yazılı anlaşma olmadıkça, Lisans kapsamında dağıtılan yazılım, herhangi bir türde (açık veya zımni) garanti veya koşul olmaksızın, "OLDUĞU GİBİ" ESASINA GÖRE dağıtılır. Lisans hükümleri, özel belirli dil kullanımı, yetkileri ve kısıtlamaları belirler.

⚠️ Bu dosya Markdown biçimindedir, ancak belge oluşturucumuz için özgü sözdizimleri içerir (MDX gibi) ve muhtemelen Markdown görüntüleyicinizde düzgün bir şekilde görüntülenmeyebilir.
-->

# 🤗 Transformers

[PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/) ve [JAX](https://jax.readthedocs.io/en/latest/) için son teknoloji makine öğrenimi.

🤗 Transformers, güncel önceden eğitilmiş (pretrained) modelleri indirmenizi ve eğitmenizi kolaylaştıran API'ler ve araçlar sunar. Önceden eğitilmiş modeller kullanarak, hesaplama maliyetlerinizi ve karbon ayak izinizi azaltabilir, ve sıfırdan bir modeli eğitmek için gereken zaman ve kaynaklardan tasarruf edebilirsiniz. Bu modeller farklı modalitelerde ortak görevleri destekler. Örneğin:

📝 **Doğal Dil İşleme**: metin sınıflandırma, adlandırılmış varlık tanıma, soru cevaplama, dil modelleme, özetleme, çeviri, çoktan seçmeli ve metin oluşturma.<br>
🖼️ **Bilgisayarlı Görü**: görüntü sınıflandırma, nesne tespiti ve bölümleme (segmentation).<br>
🗣️ **Ses**: otomatik konuşma tanıma ve ses sınıflandırma.<br>
🐙 **Çoklu Model**: tablo soru cevaplama, optik karakter tanıma, taranmış belgelerden bilgi çıkarma, video sınıflandırma ve görsel soru cevaplama.

🤗 Transformers, PyTorch, TensorFlow ve JAX arasında çerçeve (framework) uyumluluğu sağlar. Bu, bir modelin yaşam döngüsünün her aşamasında farklı bir çerçeve kullanma esnekliği sunar; bir çerçevede üç satır kodla bir modeli eğitebilir ve başka bir çerçevede tahminleme için kullanabilirsiniz. Modeller ayrıca üretim ortamlarında kullanılmak üzere ONNX ve TorchScript gibi bir formata aktarılabilir.

Büyüyen topluluğa [Hub](https://huggingface.co/models), [Forum](https://discuss.huggingface.co/) veya [Discord](https://discord.com/invite/JfAtkvEtRb) üzerinden katılabilirsiniz!

## Hugging Face ekibinden özel destek arıyorsanız

<a target="_blank" href="https://huggingface.co/support">
    <img alt="HuggingFace Uzman Hızlandırma Programı" src="https://cdn-media.huggingface.co/marketing/transformers/new-support-improved.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
</a>

## İçindekiler

Dokümantasyon, beş bölüme ayrılmıştır:

- **BAŞLARKEN**, kütüphanenin hızlı bir turunu ve çalışmaya başlamak için kurulum talimatlarını sağlar.
- **ÖĞRETİCİLER**, başlangıç yapmak için harika bir yerdir. Bu bölüm, kütüphane kullanmaya başlamak için ihtiyacınız olan temel becerileri kazanmanıza yardımcı olacaktır.
- **NASIL YAPILIR KILAVUZLARI**, önceden eğitilmiş bir modele dil modellemesi için ince ayar (fine-tuning) yapmak veya özel bir model yazmak, ve paylaşmak gibi belirli bir hedefe nasıl ulaşılacağını gösterir.
- **KAVRAMSAL REHBERLER**, modellerin, görevlerin ve 🤗 Transformers tasarım felsefesinin temel kavramları ve fikirleri hakkında daha fazla tartışma ve açıklama sunar.
- **API** tüm sınıfları (class) ve fonksiyonları (functions) açıklar:

  - **ANA SINIFLAR**, yapılandırma, model, tokenizer ve pipeline gibi en önemli sınıfları (classes) ayrıntılandırır.
  - **MODELLER**, kütüphanede kullanılan her modelle ilgili sınıfları ve fonksiyonları detaylı olarak inceler.
  - **DAHİLİ YARDIMCILAR**, kullanılan yardımcı sınıfları ve fonksiyonları detaylı olarak inceler.

## Desteklenen Modeller ve Çerçeveler

Aşağıdaki tablo, her bir model için kütüphanede yer alan mevcut desteği temsil etmektedir. Her bir model için bir Python tokenizer'ına ("slow" olarak adlandırılır) sahip olup olmadıkları, 🤗 Tokenizers kütüphanesi tarafından desteklenen hızlı bir tokenizer'a sahip olup olmadıkları, Jax (Flax aracılığıyla), PyTorch ve/veya TensorFlow'da destek olup olmadıklarını göstermektedir.

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
|                          [Fuyu](model_doc/fuyu)                          |       ✅        |         ❌         |      ❌      |
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
|                         [OWLv2](model_doc/owlv2)                         |       ✅        |         ❌         |      ❌      |
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
|                  [SeamlessM4T](model_doc/seamless_m4t)                   |       ✅        |         ❌         |      ❌      |
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
