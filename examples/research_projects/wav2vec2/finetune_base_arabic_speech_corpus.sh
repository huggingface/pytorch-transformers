#!/usr/bin/env bash
python run_asr.py \
--output_dir="./wav2vec2-base-arabic-speech-corpus" \
--num_train_epochs="30" \
--per_device_train_batch_size="24" \
--per_device_eval_batch_size="24" \
--gradient_accumulation_steps="8" \
--evaluation_strategy="steps" \
--save_steps="100" \
--eval_steps="100" \
--logging_steps="50" \
--learning_rate="5e-4" \
--warmup_steps="3000" \
--model_name_or_path="facebook/wav2vec2-base" \
--fp16 \
--dataset_name="arabic_speech_corpus" \
--train_split_name="train" \
--validation_split_name="test" \
--max_duration_in_seconds="15" \
--orthography buckwalter \
--preprocessing_num_workers="$(nproc)" \
--group_by_length \
--freeze_feature_extractor \
--target_feature_extractor_sampling_rate \
--verbose_logging \
