#!/bin/bash
# Mac-orchestrated multi-node launcher with proper synchronization
# Run this from your Mac terminal

# Configuration
NODES=("g369" "g375" "g255" "g265" "kuan-machine" "g340" "g341" "g345")
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
LAUNCH_SCRIPT="examples/qwen_image/model_training/full/launch_multinode.sh"
TMUX_SESSION_PREFIX="qwen_train"

# Get g369's actual IP
echo "Detecting main node IP..."
MAIN_IP=$(ssh g369 "hostname -I | awk '{print \$1}'")
echo "Main node (g369) IP: ${MAIN_IP}"
echo ""

echo "========================================="
echo "Launching 8-node training"
echo "Main node: g369 (${MAIN_IP})"
echo "Nodes: ${NODES[@]}"
echo "========================================="
echo ""

# Update the launch script with correct IP on all nodes
echo "Updating MAIN_IP in launch scripts on all nodes..."
for node in "${NODES[@]}"; do
    ssh ${node} "sed -i 's/MAIN_IP=.*/MAIN_IP=\"${MAIN_IP}\"/' ${WORK_DIR}/${LAUNCH_SCRIPT}" &
done
wait

echo "Creating tmux sessions on all nodes..."

# Step 1: Create all tmux sessions first
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    session_name="${TMUX_SESSION_PREFIX}_rank${i}"
    
    echo "  [Node ${i}] Creating session on ${node}..."
    ssh ${node} "tmux kill-session -t ${session_name} 2>/dev/null; tmux new-session -d -s ${session_name}" &
done
wait

echo ""
echo "Tmux sessions created. Launching training simultaneously in 3 seconds..."
sleep 3

# Step 2: Start training on ALL nodes simultaneously
echo "Starting training processes..."
for i in "${!NODES[@]}"; do
    node="${NODES[$i]}"
    session_name="${TMUX_SESSION_PREFIX}_rank${i}"
    rank=$i
    
    echo "  [Node ${rank}] Starting on ${node}..."
    ssh ${node} "tmux send-keys -t ${session_name} 'cd ${WORK_DIR} && conda activate nunchaku && bash ${LAUNCH_SCRIPT} ${rank}' C-m" &
    
    # Small delay for main node to initialize first
    if [ ${rank} -eq 0 ]; then
        sleep 3
    else
        sleep 0.1
    fi
done

wait

echo ""
echo "========================================="
echo "All nodes launched!"
echo "========================================="
echo ""
echo "Monitor training:"
echo "  Main node:   ssh g369 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank0"
echo "  Worker node: ssh g375 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank1"
echo ""
echo "Check if multi-node is working:"
echo "  Look for 'Total processes: 64' (not 8)"
echo "  Look for 'Expected sum: 2016' (not 28)"
echo ""

