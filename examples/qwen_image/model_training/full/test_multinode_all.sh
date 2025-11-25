#!/bin/bash
# Test multi-node setup on all nodes simultaneously

NODES=("g369" "g375" "g255" "g265" "g337" "g340" "g341" "g345")
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"

echo "Testing multi-node setup on all 8 nodes..."
echo ""

# Launch diagnostic on all nodes simultaneously
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    echo "Launching diagnostic on ${node} (rank ${i})..."
    
    ssh ${node} "cd ${WORK_DIR} && \
                 conda activate nunchaku && \
                 accelerate launch \
                   --config_file examples/qwen_image/model_training/full/accelerate_config_8nodes_multinode.yaml \
                   --machine_rank ${i} \
                   --main_process_ip 10.15.38.17 \
                   --main_process_port 29500 \
                   --num_machines 8 \
                   --num_processes 64 \
                   examples/qwen_image/model_training/full/diagnose_multinode.py" &
done

wait

echo ""
echo "Test complete! Check output above."
echo "If all nodes show 'Num processes: 64', multi-node is working."
echo "If nodes show 'Num processes: 8', multi-node is NOT working."

