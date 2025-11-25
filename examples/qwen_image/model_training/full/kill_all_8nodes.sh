#!/bin/bash
# Kill all training processes on 8 nodes
# Usage: ./kill_all_8nodes.sh

NODES=("g369" "g370" "g371" "g372" "g373" "g374" "g375" "g376")

echo "========================================="
echo "Killing training on all 8 nodes"
echo "========================================="
echo ""

for NODE in "${NODES[@]}"; do
    echo "Stopping training on ${NODE}..."
    ssh ${NODE} "pkill -f 'examples/qwen_image/model_training/train.py' && echo '  ✓ Killed train.py' || echo '  - No train.py running'"
    ssh ${NODE} "pkill -f 'accelerate launch' && echo '  ✓ Killed accelerate' || echo '  - No accelerate running'"
    ssh ${NODE} "pkill -f 'deepspeed' && echo '  ✓ Killed deepspeed' || echo '  - No deepspeed running'"
    echo ""
done

echo "========================================="
echo "Verifying all nodes are clean"
echo "========================================="
echo ""

for NODE in "${NODES[@]}"; do
    PROCS=$(ssh ${NODE} "ps aux | grep -E '(train\.py|accelerate launch|deepspeed)' | grep -v grep | wc -l")
    if [ "$PROCS" -eq 0 ]; then
        echo "✅ ${NODE}: Clean (no processes)"
    else
        echo "⚠️  ${NODE}: Still has ${PROCS} processes running"
        ssh ${NODE} "ps aux | grep -E '(train\.py|accelerate launch|deepspeed)' | grep -v grep"
    fi
done

echo ""
echo "Done!"

