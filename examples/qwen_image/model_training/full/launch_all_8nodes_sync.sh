#!/bin/bash
# Synchronized launcher - launches all nodes nearly simultaneously
# This gives nodes a chance to rendezvous and form a unified cluster

NODES=("g369" "g375" "g255" "g265" "g337" "g340" "g341" "g345")
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
LAUNCH_SCRIPT="examples/qwen_image/model_training/full/launch_multinode.sh"
TMUX_SESSION_PREFIX="qwen_train"

echo "========================================="
echo "Launching 8-node training with synchronized start"
echo "Nodes: ${NODES[@]}"
echo "========================================="
echo ""

# Function to launch on a single node WITHOUT waiting
launch_node_nowait() {
    local node=$1
    local rank=$2
    local session_name="${TMUX_SESSION_PREFIX}_rank${rank}"
    
    echo "[Node ${rank}] Preparing ${node}..."
    
    # Kill existing session if it exists
    ssh ${node} "tmux kill-session -t ${session_name} 2>/dev/null"
    
    # Create new session (but don't start training yet)
    ssh ${node} "tmux new-session -d -s ${session_name}"
}

# Prepare all sessions first (without starting training)
for i in "${!NODES[@]}"; do
    launch_node_nowait "${NODES[$i]}" "$i"
done

echo ""
echo "All tmux sessions created. Starting training simultaneously..."
sleep 2

# Now start training on ALL nodes at the same time
for i in "${!NODES[@]}"; do
    local node="${NODES[$i]}"
    local session_name="${TMUX_SESSION_PREFIX}_rank${i}"
    
    echo "[Node $i] Starting training on ${node}..."
    ssh ${node} "tmux send-keys -t ${session_name} 'cd ${WORK_DIR} && conda activate nunchaku && bash ${LAUNCH_SCRIPT} ${i}' C-m" &
done

wait

echo ""
echo "========================================="
echo "All nodes launched simultaneously!"
echo "========================================="
echo ""
echo "Monitor nodes:"
echo "  ssh g369 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank0"
echo "  ssh g375 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank1"

