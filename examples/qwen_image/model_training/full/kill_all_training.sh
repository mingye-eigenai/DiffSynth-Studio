#!/bin/bash
# Kill all training processes on all nodes
# Usage: ./kill_all_training.sh

NODES=("g369" "g375")  # Add more nodes as needed: ("g369" "g370" "g371" ...)

echo "Killing training on all nodes..."

for NODE in "${NODES[@]}"; do
    echo "Stopping training on ${NODE}..."
    ssh ${NODE} "pkill -f 'examples/qwen_image/model_training/train.py' || echo 'No training process found on ${NODE}'"
    ssh ${NODE} "pkill -f 'accelerate launch' || echo 'No accelerate process found on ${NODE}'"
    ssh ${NODE} "pkill -f 'deepspeed' || echo 'No deepspeed process found on ${NODE}'"
done

echo ""
echo "All nodes stopped!"
echo ""
echo "Verifying processes are gone..."
for NODE in "${NODES[@]}"; do
    echo "Checking ${NODE}:"
    ssh ${NODE} "ps aux | grep -E '(train.py|accelerate launch|deepspeed)' | grep -v grep || echo '  ✓ No training processes running'"
done

