#!/bin/bash
# FIXED Launch script for MAIN node (g369)
# This version uses the corrected DeepSpeed config with proper batch size settings

accelerate launch \
  --config_file examples/qwen_image/model_training/full/accelerate_config_multinode_FIXED.yaml \
  --machine_rank 0 \
  --main_process_ip 10.15.38.17 \
  --main_process_port 29500 \
  --num_machines 2 \
  --num_processes 16 \
  examples/qwen_image/model_training/train.py \
  --dataset_base_path data/ \
  --dataset_metadata_path /home/kuan/workspace/repos/DiffSynth-Studio/data/pico-banana-400k/openimages/metadata_sft_clean.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-Pico_2nodes_FIXED" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 10000

