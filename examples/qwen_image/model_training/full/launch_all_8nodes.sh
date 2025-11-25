#!/bin/bash
# Automated launcher for all 8 nodes via SSH with tmux
# This script launches training on all nodes in persistent tmux sessions

# Configuration
NODES=("g369" "g375" "g255" "g265" "g337" "g340" "g341" "g345")
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
LAUNCH_SCRIPT="examples/qwen_image/model_training/full/launch_multinode.sh"
TMUX_SESSION_PREFIX="qwen_train"
CONDA_ENV="nunchaku"

echo "========================================="
echo "Launching 8-node training with tmux"
echo "Nodes: ${NODES[@]}"
echo "Conda environment: ${CONDA_ENV}"
echo "========================================="
echo ""

# Function to launch on a single node
launch_node() {
    local node=$1
    local rank=$2
    local session_name="${TMUX_SESSION_PREFIX}_rank${rank}"
    
    echo "[Node ${rank}] Starting tmux session '${session_name}' on ${node}..."
    
    # Kill existing session if it exists
    ssh ${node} "tmux kill-session -t ${session_name} 2>/dev/null"
    
    # Create new session with bash shell (stays alive) and send the command
    ssh ${node} "tmux new-session -d -s ${session_name} && \
                 tmux send-keys -t ${session_name} 'cd ${WORK_DIR} && conda activate ${CONDA_ENV} && bash ${LAUNCH_SCRIPT} ${rank}' C-m"
    
    if [ $? -eq 0 ]; then
        # Verify the session was actually created
        sleep 0.5  # Brief delay for session to initialize
        ssh ${node} "tmux has-session -t ${session_name} 2>/dev/null"
        if [ $? -eq 0 ]; then
            echo "[Node ${rank}] ✓ Session created and verified on ${node}"
        else
            echo "[Node ${rank}] ✗ Session was created but closed immediately (check script path/errors)"
        fi
    else
        echo "[Node ${rank}] ✗ Failed to create tmux session on ${node}"
    fi
    
    # Small delay to ensure proper startup order
    if [ ${rank} -eq 0 ]; then
        echo "[Node 0] Main node starting, waiting 5 seconds for initialization..."
        sleep 5
    else
        sleep 1
    fi
}

# Launch on all nodes
for i in "${!NODES[@]}"; do
    launch_node "${NODES[$i]}" "$i"
done

echo ""
echo "========================================="
echo "All tmux sessions launched!"
echo "========================================="
echo ""
echo "To monitor a specific node:"
echo "  ssh <node> -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank<N>"
echo ""
echo "Examples:"
echo "  ssh g369 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank0"
echo "  ssh g370 -t tmux attach -t ${TMUX_SESSION_PREFIX}_rank1"
echo ""
echo "To list all sessions on a node:"
echo "  ssh <node> tmux ls"
echo ""
echo "To kill all training sessions:"
echo "  for node in ${NODES[@]}; do ssh \$node tmux kill-session -t ${TMUX_SESSION_PREFIX}_rank*; done"
echo ""

