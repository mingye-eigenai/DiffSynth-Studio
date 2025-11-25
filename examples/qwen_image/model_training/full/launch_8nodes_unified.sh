#!/bin/bash
# Unified multi-node launcher - Run ONLY on main node (g369)
# This script launches training across all nodes from the main node

WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
CONFIG_FILE="${WORK_DIR}/examples/qwen_image/model_training/full/accelerate_config_8nodes_multinode.yaml"

echo "========================================="
echo "Launching 8-node training from main node"
echo "Using config: ${CONFIG_FILE}"
echo "========================================="
echo ""

cd ${WORK_DIR}

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nunchaku

# Disable ModelScope auto-download/checking
export MODELSCOPE_CACHE=/home/kuan/.cache/modelscope
export MODELSCOPE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Launch using Accelerate with proper multi-node setup
accelerate launch \
  --config_file ${CONFIG_FILE} \
  --num_processes 64 \
  --num_machines 8 \
  --machine_rank 0 \
  --main_process_ip 10.15.38.17 \
  --main_process_port 29500 \
  --rdzv_backend static \
  --same_network \
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

