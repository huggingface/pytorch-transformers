<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

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

# Performance e Scalabilità

Allenare grandi modelli transformer e metterli in produzione presenta diverse sfide. Durante l’allenamento, il modello potrebbe richiedere più memoria GPU di quella disponibile o avere una velocità di allenamento lenta. Nella fase di deployment, il modello può avere difficoltà a gestire il throughput necessario in un ambiente di produzione.

Questa documentazione ha l’obiettivo di aiutarti a superare queste sfide e trovare le impostazioni ottimali per il tuo caso d’uso. Le guide sono divise in sezioni di allenamento e inferenza, poiché ognuna presenta sfide e soluzioni diverse. All'interno di ciascuna sezione troverai guide separate per diverse configurazioni hardware, come singola GPU vs. multi-GPU per l'allenamento o CPU vs. GPU per l'inferenza.

Usa questo documento come punto di partenza per navigare verso i metodi che si adattano al tuo scenario.

## Allenamento

Allenare grandi modelli transformer in modo efficiente richiede un acceleratore come una GPU o una TPU. Il caso più comune è avere una singola GPU. I metodi che puoi applicare per migliorare l’efficienza dell'allenamento su una singola GPU si estendono anche ad altri setup come le GPU multiple. Tuttavia, ci sono anche tecniche specifiche per l’allenamento su multi-GPU o CPU. Ne parliamo in sezioni separate.

* [Metodi e strumenti per un allenamento efficiente su una singola GPU](perf_train_gpu_one): inizia qui per scoprire approcci comuni che possono aiutarti a ottimizzare l’utilizzo della memoria GPU, ad accelerare l’allenamento, o entrambi. 
* [Sezione di allenamento multi-GPU](perf_train_gpu_many): esplora questa sezione per conoscere metodi di ottimizzazione ulteriori che si applicano a un setting multi-GPU, come il parallelismo dei dati, dei tensori e delle pipeline.
* [Sezione di allenamento su CPU](perf_train_cpu): scopri l’allenamento a precisione mista su CPU.
* [Allenamento efficiente su più CPU](perf_train_cpu_many): scopri l'allenamento distribuito su CPU.
* [Allenamento su TPU con TensorFlow](perf_train_tpu_tf): se sei nuovo con le TPU, fai riferimento a questa sezione per un’introduzione orientata all’allenamento su TPU e all'uso di XLA. 
* [Hardware personalizzato per l'allenamento](perf_hardware): trova suggerimenti e trucchi per costruire il tuo rig di deep learning.
* [Ricerca di iperparametri usando Trainer API](hpo_train)

## Inferenza

L’inferenza efficiente con grandi modelli in un ambiente di produzione può essere sfidante quanto il loro allenamento. Nelle sezioni seguenti vediamo i passaggi per eseguire l'inferenza su setup CPU e singola/multi-GPU.

* [Inferenza su una singola CPU](perf_infer_cpu)
* [Inferenza su una singola GPU](perf_infer_gpu_one)
* [Inferenza multi-GPU](perf_infer_gpu_one)
* [Integrazione XLA per modelli TensorFlow](tf_xla)

## Allenamento e inferenza

Qui troverai tecniche, suggerimenti e trucchi che si applicano sia che tu stia allenando un modello, sia che stia eseguendo inferenza con esso.

* [Istituire un grande modello](big_models)
* [Risoluzione dei problemi delle performance](debugging)

## Contribuire

Questo documento è lontano dall'essere completo e c'è molto altro da aggiungere, quindi se hai aggiunte o correzioni da fare non esitare ad aprire una PR oppure, se non sei sicuro, inizia un'Issue e possiamo discutere i dettagli lì.

Quando fai delle contribuzioni che A è meglio di B, cerca di includere un benchmark riproducibile e/o un link alla fonte di quelle informazioni (a meno che non venga direttamente da te).
