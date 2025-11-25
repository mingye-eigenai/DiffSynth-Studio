# Use local model paths to avoid downloading
MODEL_BASE_PATH="/shared/mingye/DiffSynth-Studio/examples/qwen_image/model_training/lora/models/Qwen"

accelerate launch /shared/mingye/DiffSynth-Studio/examples/qwen_image/model_training/train_merged_qkv.py \
  --dataset_base_path /shared/mingye/pico_dataset \
  --dataset_metadata_path /shared/mingye/sft_with_local_paths_qwen_new.csv \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths "[\"${MODEL_BASE_PATH}/Qwen-Image-Edit/transformer/diffusion_pytorch_model*.safetensors\",\"${MODEL_BASE_PATH}/Qwen-Image/text_encoder/model*.safetensors\",\"${MODEL_BASE_PATH}/Qwen-Image/vae/diffusion_pytorch_model.safetensors\"]" \
  --tokenizer_path "${MODEL_BASE_PATH}/Qwen-Image/tokenizer" \
  --processor_path "${MODEL_BASE_PATH}/Qwen-Image-Edit/processor" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/shared/mingye/DiffSynth-Studio/examples/qwen_image/model_training/lora/models/train/Qwen_lora_rank32_pico_kuan" \
  --lora_base_model "dit" \
  --lora_target_modules "to_qkv,add_qkv_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,txt_mlp.net.0.proj,txt_mlp.net.2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 3000

# Note: This training script uses the efficient merged QKV architecture where:
# - to_q, to_k, to_v are merged into to_qkv (reduces kernel overhead)
# - add_q_proj, add_k_proj, add_v_proj are merged into add_qkv_proj
# 
# The merged architecture provides:
# 1. Reduced kernel call overhead during forward pass
# 2. More efficient memory access patterns
# 3. Faster training and inference
#
# LoRA target modules have been updated to reflect the merged architecture:
# - Removed: to_q, to_k, to_v, add_q_proj, add_k_proj, add_v_proj
# - Added: to_qkv, add_qkv_proj
#
# The trained LoRA can be converted back to standard format if needed using the
# state dict converter.

# --lora_fused "./models/lightx2v/Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors" \

# Original separate QKV target modules (for reference):
# --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \

