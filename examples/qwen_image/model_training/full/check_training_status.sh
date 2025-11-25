#!/bin/bash
# Check training status on all nodes
# Usage: ./check_training_status.sh [num_nodes]

NUM_NODES=${1:-2}  # Default to 2 nodes if not specified

if [ "$NUM_NODES" -eq 2 ]; then
    NODES=("g369" "g375")
elif [ "$NUM_NODES" -eq 8 ]; then
    NODES=("g369" "g370" "g371" "g372" "g373" "g374" "g375" "g376")
else
    echo "Error: Only 2 or 8 nodes supported"
    echo "Usage: $0 [2|8]"
    exit 1
fi

echo "========================================="
echo "Training Status Check - ${NUM_NODES} Nodes"
echo "========================================="
echo ""

TOTAL_PROCS=0

for NODE in "${NODES[@]}"; do
    echo "Checking ${NODE}:"
    
    # Check for training processes
    TRAIN_PROCS=$(ssh ${NODE} "ps aux | grep 'train.py' | grep -v grep | wc -l")
    ACCEL_PROCS=$(ssh ${NODE} "ps aux | grep 'accelerate launch' | grep -v grep | wc -l")
    
    if [ "$TRAIN_PROCS" -gt 0 ] || [ "$ACCEL_PROCS" -gt 0 ]; then
        echo "  ✅ RUNNING (train: ${TRAIN_PROCS}, accelerate: ${ACCEL_PROCS})"
        
        # Show GPU usage
        GPU_INFO=$(ssh ${NODE} "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null" || echo "GPU info unavailable")
        if [ "$GPU_INFO" != "GPU info unavailable" ]; then
            echo "  GPU Usage:"
            echo "$GPU_INFO" | while read line; do
                echo "    GPU $line"
            done
        fi
        
        TOTAL_PROCS=$((TOTAL_PROCS + TRAIN_PROCS))
    else
        echo "  ⭕ IDLE (no training)"
    fi
    echo ""
done

echo "========================================="
echo "Summary: ${TOTAL_PROCS} training process(es) across ${NUM_NODES} nodes"
echo "========================================="

if [ "$TOTAL_PROCS" -eq 0 ]; then
    echo "Status: No training running"
elif [ "$TOTAL_PROCS" -eq "$NUM_NODES" ]; then
    echo "Status: ✅ All nodes running normally"
else
    echo "Status: ⚠️  WARNING: Partial training (${TOTAL_PROCS}/${NUM_NODES} nodes)"
    echo "This may indicate a problem. Check logs or kill all processes."
fi

