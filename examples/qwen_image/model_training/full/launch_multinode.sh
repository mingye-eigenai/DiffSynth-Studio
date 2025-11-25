#!/bin/bash
# Universal launch script for multi-node training
# Usage: ./launch_multinode.sh <machine_rank>
# Example: 
#   On node 0 (main): ./launch_multinode.sh 0
#   On node 1: ./launch_multinode.sh 1
#   On node 2: ./launch_multinode.sh 2
#   ... and so on

if [ -z "$1" ]; then
    echo "Error: Please provide machine_rank as argument"
    echo "Usage: $0 <machine_rank>"
    echo "Example: $0 0  (for main node)"
    echo "         $0 1  (for worker node 1)"
    exit 1
fi

MACHINE_RANK=$1
MAIN_IP="10.15.38.17"
MAIN_PORT="29500"
NUM_MACHINES=8
NUM_PROCESSES=64  # 8 GPUs per node * 8 nodes
NUM_GPUS_PER_NODE=8

echo "============================================="
echo "Node ${MACHINE_RANK}/${NUM_MACHINES} Starting"
echo "============================================="
echo "  Main process: ${MAIN_IP}:${MAIN_PORT}"
echo "  Total processes: ${NUM_PROCESSES}"
echo "  This node IP: $(hostname -I | awk '{print $1}')"
echo "  Can reach main? $(ping -c 1 -W 1 ${MAIN_IP} >/dev/null 2>&1 && echo 'YES' || echo 'NO')"
echo "============================================="

# Set environment variables to ensure proper multi-node coordination
export WORLD_SIZE=${NUM_PROCESSES}
export MASTER_ADDR=${MAIN_IP}
export MASTER_PORT=${MAIN_PORT}
export NODE_RANK=${MACHINE_RANK}
export RANK=$((${MACHINE_RANK} * ${NUM_GPUS_PER_NODE}))
export LOCAL_RANK=0

# Disable ModelScope auto-download/checking to avoid lock contention
export MODELSCOPE_CACHE=/home/kuan/.cache/modelscope
export MODELSCOPE_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Launch without config file, specify everything via CLI
accelerate launch \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file examples/qwen_image/model_training/full/ds_config_8nodes.json \
  --machine_rank ${MACHINE_RANK} \
  --main_process_ip ${MAIN_IP} \
  --main_process_port ${MAIN_PORT} \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${NUM_PROCESSES} \
  --gradient_accumulation_steps 8 \
  --zero_stage 2 \
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

