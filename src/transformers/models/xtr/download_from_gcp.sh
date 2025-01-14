#!/usr/bin/env bash
# Use this script as follows ./download_from_gcp.sh /path/to/folder/to/store/downloads
folder_to_store_downloads=${1}

# Replace by gcp_path to T5 cloud bucket folder here
# To download the official `t5-small` model of https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints:
gcp_path="gs://t5-data/pretrained_models/small"

# Number of files the checkpoint is split into
num_of_checks=16

# Create dir if not exist
mkdir -p ${folder_to_store_downloads}

# Copy all meta information files
gsutil cp "${gcp_path}/operative_config.gin" ${folder_to_store_downloads}
gsutil cp "${gcp_path}/checkpoint" ${folder_to_store_downloads}
gsutil cp "${gcp_path}/model.ckpt-1000000.index" ${folder_to_store_downloads}
gsutil cp "${gcp_path}/model.ckpt-1000000.meta" ${folder_to_store_downloads}

# Copy all model weights
# single digit num checkpoitns
for ((i = 0 ; i < ${num_of_checks} ; i++)); do
	gsutil cp "${gcp_path}/model.ckpt-1000000.data-0000${i}-of-000${num_of_checks}" ${folder_to_store_downloads}
done

# double digit num checkpoints
for ((i = 0 ; i < ${num_of_checks} ; i++)); do
	gsutil cp "${gcp_path}/model.ckpt-1000000.data-000${i}-of-000${num_of_checks}" ${folder_to_store_downloads}
done


# Having run this script, you should create a suitable config.json, *e.g.* by 
# looking at `https://huggingface.co/t5-small`.
# Then you can run `python convert_t5_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path "${folder_to_store_downloads}" --config_file "config.json" --pytorch_dump_path "/path/to/store/pytorch/weights"
