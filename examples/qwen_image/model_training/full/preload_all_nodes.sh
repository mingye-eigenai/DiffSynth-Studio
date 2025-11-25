#!/bin/bash
# Run model pre-loading on all nodes

NODES=("g369" "g375" "g255" "g265" "g337" "g340" "g341" "g345")
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"

echo "Pre-loading models on all 8 nodes..."

for node in "${NODES[@]}"; do
    echo "Starting pre-load on ${node}..."
    ssh ${node} "cd ${WORK_DIR} && bash examples/qwen_image/model_training/full/preload_models.sh" &
done

wait

echo ""
echo "✅ All nodes have models pre-loaded!"

