#!/bin/bash
# DeepSpeed multi-node launcher using hostfile
# Run this ONLY on the main node (g369)

WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
HOSTFILE="${WORK_DIR}/examples/qwen_image/model_training/full/hostfile_8nodes"

echo "========================================="
echo "Launching 8-node DeepSpeed training"
echo "Using hostfile: ${HOSTFILE}"
echo "========================================="
echo ""

cd ${WORK_DIR}

# Activate conda environment
conda activate nunchaku

# Launch with DeepSpeed
deepspeed --hostfile=${HOSTFILE} \
  --master_addr=10.15.38.17 \
  --master_port=29500 \
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
  --output_path "./models/train/Qwen-Image-Edit-Pico_8nodes" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 1000

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="

