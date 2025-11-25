#!/bin/bash
# Automated launcher for 2-node training with proper cleanup and monitoring
# This script handles everything automatically

set -e  # Exit on error

MAIN_NODE="g369"
WORKER_NODE="g375"
WORK_DIR="/home/kuan/workspace/repos/DiffSynth-Studio"
LOG_DIR="${WORK_DIR}/logs/multinode_training_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Automated 2-Node Training Launcher"
echo "Main: ${MAIN_NODE}, Worker: ${WORKER_NODE}"
echo "========================================="
echo ""

# Step 1: Clean up existing processes
echo "Step 1: Cleaning up existing training processes..."
${WORK_DIR}/examples/qwen_image/model_training/full/kill_all_training.sh
sleep 3

# Step 2: Create log directory
echo ""
echo "Step 2: Creating log directory..."
mkdir -p ${LOG_DIR}
echo "Logs will be saved to: ${LOG_DIR}"

# Step 3: Launch main node
echo ""
echo "Step 3: Launching main node (${MAIN_NODE})..."
ssh ${MAIN_NODE} "cd ${WORK_DIR} && \
    nohup bash examples/qwen_image/model_training/full/launch_main_g369_FIXED.sh \
    > ${LOG_DIR}/main_g369.log 2>&1 &"

echo "Main node launched! Waiting 10 seconds for initialization..."
sleep 10

# Step 4: Check if main node started successfully
echo ""
echo "Step 4: Verifying main node started..."
MAIN_PROC=$(ssh ${MAIN_NODE} "ps aux | grep 'train.py' | grep -v grep | wc -l")
if [ "$MAIN_PROC" -eq 0 ]; then
    echo "❌ ERROR: Main node failed to start!"
    echo "Check logs: ${LOG_DIR}/main_g369.log"
    exit 1
else
    echo "✅ Main node is running"
fi

# Step 5: Launch worker node
echo ""
echo "Step 5: Launching worker node (${WORKER_NODE})..."
ssh ${WORKER_NODE} "cd ${WORK_DIR} && \
    nohup bash examples/qwen_image/model_training/full/launch_worker_g375_FIXED.sh \
    > ${LOG_DIR}/worker_g375.log 2>&1 &"

echo "Worker node launched! Waiting 5 seconds..."
sleep 5

# Step 6: Verify both nodes are running
echo ""
echo "Step 6: Final verification..."
${WORK_DIR}/examples/qwen_image/model_training/full/check_training_status.sh 2

# Step 7: Show log locations and monitoring commands
echo ""
echo "========================================="
echo "✅ Training Started Successfully!"
echo "========================================="
echo ""
echo "Log files:"
echo "  Main:   ${LOG_DIR}/main_g369.log"
echo "  Worker: ${LOG_DIR}/worker_g375.log"
echo ""
echo "Monitoring commands:"
echo "  Check status:     ./examples/qwen_image/model_training/full/check_training_status.sh 2"
echo "  View main log:    tail -f ${LOG_DIR}/main_g369.log"
echo "  View worker log:  tail -f ${LOG_DIR}/worker_g375.log"
echo "  Stop training:    ./examples/qwen_image/model_training/full/kill_all_training.sh"
echo ""
echo "Training is running in background. You can safely close this terminal."

