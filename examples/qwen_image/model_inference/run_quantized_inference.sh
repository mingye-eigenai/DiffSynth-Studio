#!/bin/bash

# Script to run quantized inference with different modes

echo "Qwen-Image Quantized Inference Runner"
echo "====================================="
echo ""
echo "Select optimization mode:"
echo "0) H100-OPTIMIZED with Flash Attention (FASTEST for H100)"
echo "1) torch.compile() Optimization (Good without Flash Attn)"
echo "2) FP8 Quantized (Standard - Memory savings)"
echo "3) FP8 Quantized OPTIMIZED with LoRA Fusion"
echo "4) Low VRAM Mode (Standard - 8GB VRAM)"
echo "5) Low VRAM Mode OPTIMIZED with LoRA Fusion (6GB+ VRAM)"
echo "6) Benchmark quantization modes"
echo "7) Benchmark LoRA loading strategies"
echo "8) Diagnose FP8 performance issues"
echo "9) Check GPU FP8 support"
echo "i) Install Flash Attention for H100"
echo ""
read -p "Enter your choice (0-9 or i): " choice

case $choice in
    0)
        echo "Running H100-OPTIMIZED inference with Flash Attention..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-H100-Optimized.py
        ;;
    1)
        echo "Running torch.compile() optimized inference..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-TorchCompile.py
        ;;
    2)
        echo "Running FP8 Quantized inference..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-Quantized.py
        ;;
    3)
        echo "Running OPTIMIZED FP8 Quantized inference with LoRA fusion..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-Quantized-Optimized.py
        ;;
    4)
        echo "Running Low VRAM mode..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-LowVRAM.py
        ;;
    5)
        echo "Running OPTIMIZED Low VRAM mode with LoRA fusion..."
        python examples/qwen_image/model_inference/Qwen-Image-Edit-LoRA-Ghost-LowVRAM-Optimized.py
        ;;
    6)
        echo "Running quantization benchmark..."
        python examples/qwen_image/model_inference/compare_quantization.py
        ;;
    7)
        echo "Running LoRA loading strategy benchmark..."
        python examples/qwen_image/model_inference/benchmark_lora_speed.py
        ;;
    8)
        echo "Diagnosing FP8 performance..."
        python examples/qwen_image/model_inference/diagnose_fp8_performance.py
        ;;
    9)
        echo "Checking GPU FP8 support..."
        python examples/qwen_image/model_inference/check_fp8_support.py
        ;;
    i|I)
        echo "Installing Flash Attention for H100..."
        bash install_flash_attention.sh
        ;;
    *)
        echo "Invalid choice. Please run again and select 0-9 or i."
        exit 1
        ;;
esac

echo ""
echo "Done!"
