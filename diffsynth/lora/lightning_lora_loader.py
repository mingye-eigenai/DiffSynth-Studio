import torch


class LightningLoRALoader:
    """Loader for Lightning LoRA format that uses lora_down/lora_up naming"""
    
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
    
    def get_name_dict(self, lora_state_dict):
        """Extract layer names from Lightning LoRA format"""
        lora_name_dict = {}
        for key in lora_state_dict:
            if "lora_up.weight" not in key:
                continue
            
            # Extract the base layer name by removing the lora suffix
            base_key = key.replace(".lora_up.weight", "")
            lora_down_key = key.replace(".lora_up.weight", ".lora_down.weight")
            
            # Check if the corresponding lora_down exists
            if lora_down_key in lora_state_dict:
                lora_name_dict[base_key] = (key, lora_down_key)
        
        return lora_name_dict
    
    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        """Apply Lightning LoRA weights to model"""
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        
        # Get alpha values if they exist
        alpha_dict = {}
        for key in state_dict_lora:
            if key.endswith(".alpha"):
                base_key = key.replace(".alpha", "")
                alpha_dict[base_key] = state_dict_lora[key].item()
        
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up_key, weight_down_key = lora_name_dict[name]
                weight_up = state_dict_lora[weight_up_key].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[weight_down_key].to(device=self.device, dtype=self.torch_dtype)
                
                # Get alpha for this layer if it exists
                # layer_alpha = alpha_dict.get(name, alpha * weight_up.shape[-1]) / weight_up.shape[-1]
                layer_alpha = alpha * alpha_dict.get(name, weight_up.shape[-1]) / weight_up.shape[-1]
                # print(f"Layer {name} alpha: {layer_alpha}")
                
                # Compute LoRA weight update
                if len(weight_up.shape) == 4:
                    # Conv2d case
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = layer_alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    # Linear case
                    weight_lora = layer_alpha * torch.mm(weight_up, weight_down)
                
                # Apply update to module weights
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        
        print(f"{updated_num} tensors are updated by Lightning LoRA.")
        return updated_num
