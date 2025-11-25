import torch, os, json, glob
from diffsynth import load_state_dict
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.pipelines.flux_image_new import ControlNetInput
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, qwen_image_parser, launch_training_task, launch_data_process_task
from diffsynth.trainers.unified_dataset import UnifiedDataset
from diffsynth.models.qwen_image_dit_merged_qkv import QwenImageDiTMergedQKV
from diffsynth.models import ModelManager
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging for tracking skipped data
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper dataset that handles missing or corrupted data gracefully.
    When an error occurs loading data, it skips to the next valid item.
    """
    def __init__(self, base_dataset, max_retries=10):
        self.base_dataset = base_dataset
        self.max_retries = max_retries
        self.error_count = 0
        self.error_indices = set()

    def __len__(self):
        return len(self.base_dataset)

    def __getattr__(self, name):
        """Forward attribute access to the base dataset"""
        return getattr(self.base_dataset, name)

    def __getitem__(self, idx):
        """
        Try to get item at idx. If it fails, try the next items until success.
        """
        for retry in range(self.max_retries):
            try:
                current_idx = (idx + retry) % len(self.base_dataset)
                data = self.base_dataset[current_idx]

                # Validate that critical data is present
                if data is None:
                    raise ValueError(f"Data at index {current_idx} is None")

                return data

            except Exception as e:
                current_idx = (idx + retry) % len(self.base_dataset)

                # Log error only once per index
                if current_idx not in self.error_indices:
                    self.error_count += 1
                    self.error_indices.add(current_idx)
                    logger.warning(
                        f"Failed to load data at index {current_idx} (retry {retry+1}/{self.max_retries}): {type(e).__name__}: {str(e)}"
                    )

                # If this is the last retry, raise the exception
                if retry == self.max_retries - 1:
                    logger.error(
                        f"Failed to load data after {self.max_retries} retries starting from index {idx}. "
                        f"Total errors encountered: {self.error_count}"
                    )
                    raise RuntimeError(
                        f"Unable to load valid data after {self.max_retries} attempts. "
                        f"Please check your dataset for corrupted files."
                    ) from e

                # Otherwise, continue to next item
                continue

        # This should never be reached due to the raise in the loop
        raise RuntimeError("Unexpected state in RobustDatasetWrapper")


class QwenImageMergedQKVTrainingModule(DiffusionTrainingModule):
    """
    Training module for Qwen-Image-Edit with merged QKV projections.
    This module uses QwenImageDiTMergedQKV which merges to_q, to_k, to_v into to_qkv
    and add_q_proj, add_k_proj, add_v_proj into add_qkv_proj for efficiency.
    """
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None, lora_fused=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        enable_fp8_training=False,
        task="sft",
    ):
        super().__init__()
        
        # Load models with merged QKV architecture
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=enable_fp8_training)
        
        # Replace the standard DiT with merged QKV version
        # First, we need to find and modify the DiT config
        dit_config = None
        for i, config in enumerate(model_configs):
            # Check origin_file_pattern if it exists, or check path for transformer/dit
            is_transformer = False
            if config.origin_file_pattern is not None and "transformer" in config.origin_file_pattern.lower():
                is_transformer = True
            elif config.path is not None and ("transformer" in str(config.path).lower() or "dit" in str(config.path).lower()):
                is_transformer = True

            if is_transformer:
                dit_config = config
                break
        
        # Load the pipeline with standard components first
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        
        # Initialize pipeline
        self.pipe = QwenImagePipeline(torch_dtype=torch.bfloat16, device="cpu")
        
        # Load non-DiT models normally
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()

            # Expand glob patterns in paths
            if model_config.path is not None and isinstance(model_config.path, str) and ('*' in model_config.path or '?' in model_config.path):
                expanded_paths = sorted(glob.glob(model_config.path))
                if expanded_paths:
                    model_config.path = expanded_paths
                    logger.info(f"Expanded wildcard path to {len(expanded_paths)} files: {model_config.path[0]} ...")

            # Skip DiT loading here - we'll handle it separately
            is_transformer = False
            if model_config.origin_file_pattern is not None and "transformer" in model_config.origin_file_pattern.lower():
                is_transformer = True
            elif model_config.path is not None and ("transformer" in str(model_config.path).lower() or "dit" in str(model_config.path).lower()):
                is_transformer = True

            if not is_transformer:
                logger.info(f"Loading models from: {model_config.path}")
                model_manager.load_model(
                    model_config.path,
                    device=model_config.offload_device or "cpu",
                    torch_dtype=model_config.offload_dtype or torch.bfloat16
                )
        
        # Load text encoder and VAE
        self.pipe.text_encoder = model_manager.fetch_model("qwen_image_text_encoder")
        self.pipe.vae = model_manager.fetch_model("qwen_image_vae")
        
        # Load tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            self.pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            self.pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        
        # Load DiT with merged QKV architecture
        print("Loading DiT with merged QKV architecture...")
        if dit_config is not None:
            dit_config.download_if_necessary()

            # Expand glob patterns in DiT path
            if dit_config.path is not None and isinstance(dit_config.path, str) and ('*' in dit_config.path or '?' in dit_config.path):
                expanded_paths = sorted(glob.glob(dit_config.path))
                if expanded_paths:
                    dit_config.path = expanded_paths
                    logger.info(f"Expanded DiT wildcard path to {len(expanded_paths)} files")

            # Load pretrained weights FIRST
            # Handle both single file and sharded models
            dit_path = dit_config.path
            if isinstance(dit_path, list):
                # Sharded model - load all shards
                print(f"Loading sharded DiT from {len(dit_path)} files...")
                state_dict = {}
                for shard_path in dit_path:
                    shard_dict = load_state_dict(shard_path)
                    state_dict.update(shard_dict)
            else:
                # Single file
                state_dict = load_state_dict(dit_path)
            
            print(f"Loaded {len(state_dict)} parameters from DiT")
            
            # Create the merged QKV DiT model
            dit_merged = QwenImageDiTMergedQKV(num_layers=60)
            
            # Convert to merged format
            converter = dit_merged.state_dict_converter()
            merged_state_dict = converter.from_diffusers(state_dict)
            
            print(f"Converted to merged format: {len(merged_state_dict)} parameters")
            
            # CRITICAL: Convert model to bfloat16 BEFORE loading bfloat16 weights
            # This preserves weights exactly without dtype conversion
            dit_merged = dit_merged.to(dtype=torch.bfloat16, device="cpu")
            
            # NOW load the bfloat16 weights into bfloat16 model
            missing, unexpected = dit_merged.load_state_dict(merged_state_dict, strict=False)
            
            if missing:
                print(f"Missing keys: {len(missing)} (first 5): {missing[:5]}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
            if len(missing) == 0 and len(unexpected) == 0:
                print("✓ All weights loaded successfully!")
            
            self.pipe.dit = dit_merged
            print("✓ Successfully loaded DiT with merged QKV architecture")
        else:
            raise ValueError("No DiT configuration found in model_configs")

        # Fuse pretrained LoRA into base weights if provided
        if lora_fused is not None:
            print(f"Loading and fusing pretrained LoRA from: {lora_fused}")
            lora_state_dict = load_state_dict(lora_fused)
            
            # Check if this is a Lightning LoRA
            is_lightning_lora = any("lora_down" in k or "lora_up" in k for k in lora_state_dict.keys())
            
            if is_lightning_lora:
                from diffsynth.lora.lightning_lora_loader import LightningLoRALoader
                lora_loader = LightningLoRALoader(torch_dtype=torch.bfloat16, device="cpu")
            else:
                from diffsynth.lora import GeneralLoRALoader
                lora_loader = GeneralLoRALoader(torch_dtype=torch.bfloat16, device="cpu")
            
            if lora_base_model is not None:
                lora_loader.load(getattr(self.pipe, lora_base_model), lora_state_dict, alpha=1.0)
            else:
                lora_loader.load(self.pipe.dit, lora_state_dict, alpha=1.0)
            print("LoRA fused into base weights successfully")

        # Training mode - update lora_target_modules to use merged QKV layers
        if lora_target_modules:
            # Replace separate Q, K, V modules with merged modules
            original_modules = lora_target_modules.split(",")
            merged_modules = []
            
            # Skip individual Q, K, V projections as they're now merged
            skip_modules = {"to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"}
            for module in original_modules:
                if module.strip() not in skip_modules:
                    merged_modules.append(module.strip())
            
            # Add merged QKV modules
            merged_modules.extend(["to_qkv", "add_qkv_proj"])
            lora_target_modules = ",".join(merged_modules)
            print(f"Updated LoRA target modules for merged QKV: {lora_target_modules}")
        
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=enable_fp8_training,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task

    
    def forward_preprocess(self, data):
        try:
            # CFG-sensitive parameters
            inputs_posi = {"prompt": data["prompt"]}
            inputs_nega = {"negative_prompt": ""}

            # CFG-unsensitive parameters
            inputs_shared = {
                # Assume you are using this pipeline for inference,
                # please fill in the input parameters.
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
                # Please do not modify the following parameters
                # unless you clearly know what this will cause.
                "cfg_scale": 1,
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "edit_image_auto_resize": True,
            }

            # Extra inputs
            controlnet_input, blockwise_controlnet_input = {}, {}
            for extra_input in self.extra_inputs:
                if extra_input.startswith("blockwise_controlnet_"):
                    blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
                elif extra_input.startswith("controlnet_"):
                    controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
                else:
                    inputs_shared[extra_input] = data[extra_input]
            if len(controlnet_input) > 0:
                inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
            if len(blockwise_controlnet_input) > 0:
                inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]

            # Pipeline units will automatically process the input parameters.
            for unit in self.pipe.units:
                inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
            return {**inputs_shared, **inputs_posi}

        except Exception as e:
            logger.error(f"Error in forward_preprocess: {type(e).__name__}: {str(e)}")
            logger.error(f"Problematic data keys: {list(data.keys())}")
            raise
    
    
    def forward(self, data, inputs=None, return_inputs=False):
        # Inputs
        if inputs is None:
            inputs = self.forward_preprocess(data)
        else:
            inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        if return_inputs: return inputs
        
        # Loss
        if self.task == "sft":
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            loss = self.pipe.training_loss(**models, **inputs)
        elif self.task == "data_process":
            loss = inputs
        elif self.task == "direct_distill":
            loss = self.pipe.direct_distill_loss(**inputs)
        else:
            raise NotImplementedError(f"Unsupported task: {self.task}.")
        return loss


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()

    # Create base dataset
    base_dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )

    # Wrap dataset with robust error handling
    logger.info("=" * 80)
    logger.info("Dataset Configuration:")
    logger.info(f"  Base dataset size: {len(base_dataset)}")
    logger.info(f"  Wrapped with RobustDatasetWrapper (max_retries=10)")
    logger.info(f"  If data loading fails, training will skip to the next valid item")
    logger.info("=" * 80)
    dataset = RobustDatasetWrapper(base_dataset, max_retries=10)

    model = QwenImageMergedQKVTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lora_fused=args.lora_fused,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        enable_fp8_training=args.enable_fp8_training,
        task=args.task,
    )
    model_logger = ModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    launcher_map = {
        "sft": launch_training_task,
        "data_process": launch_data_process_task,
        "direct_distill": launch_training_task,
    }
    launcher_map[args.task](dataset, model, model_logger, args=args)

