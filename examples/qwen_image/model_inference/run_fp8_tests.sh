#!/bin/bash
# Run FP8 vs BF16 performance tests

echo "🔬 Running FP8 vs BF16 Performance Tests"
echo "========================================"

# Check GPU
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Test 1: Linear layer microbenchmark
echo -e "\n📊 Test 1: Linear Layer Microbenchmark"
echo "Testing isolated linear layer performance..."
python test_fp8_vs_bf16_linear.py

# Test 2: Full model benchmark
echo -e "\n🚀 Test 2: Full Model Inference Benchmark"
echo "Testing complete model inference performance..."
python test_fp8_vs_bf16_full_model.py

echo -e "\n✅ All tests complete!"
echo "Results saved in:"
echo "  - ./fp8_benchmark_results/      (linear layer results)"
echo "  - ./fp8_vs_bf16_results/        (full model results)"
echo "  - ./test_outputs/               (generated images)"
