# Multi-Node Training Management Guide

## Why Killing One Node Doesn't Stop Others

### The Behavior You're Experiencing

```bash
# Scenario 1: Kill main node
ssh g369
Ctrl+C  # Kills training on g369
exit

# Check g375
ssh g375
ps aux | grep train.py  # ❌ STILL RUNNING! Process is hanging
```

```bash
# Scenario 2: Kill worker node
ssh g375
Ctrl+C  # Kills training on g375
exit

# Check g369
ssh g369
ps aux | grep train.py  # ❌ STILL RUNNING! Process is hanging
```

### Why This Happens

1. **Independent Processes**: Each node runs its own Python process
2. **No Auto-Termination**: Processes don't monitor each other's status
3. **Network Waiting**: The surviving node(s) wait indefinitely for the killed node
4. **Hung State**: Eventually the process may error out, but it won't cleanly exit

This is **standard behavior** in distributed training frameworks (DeepSpeed, PyTorch DDP, etc.).

---

## Proper Multi-Node Management

### ✅ Method 1: Kill All Nodes at Once (Recommended)

**For 2-node training:**
```bash
./examples/qwen_image/model_training/full/kill_all_training.sh
```

**For 8-node training:**
```bash
./examples/qwen_image/model_training/full/kill_all_8nodes.sh
```

This will:
- SSH into each node
- Kill all training-related processes
- Verify everything is stopped
- Report status

### ✅ Method 2: Use `tmux` or `screen` for Better Control

Launch each node in a tmux session:

```bash
# On g369 (in tmux):
tmux new -s training_main
./examples/qwen_image/model_training/full/launch_main_g369.sh
# Detach with Ctrl+B, then D

# On g375 (in tmux):
tmux new -s training_worker
./examples/qwen_image/model_training/full/launch_worker_g375.sh
# Detach with Ctrl+B, then D
```

To stop everything:
```bash
# On each node
tmux attach -t training_main  # or training_worker
Ctrl+C  # Then do this on BOTH nodes
```

Or kill tmux sessions remotely:
```bash
ssh g369 "tmux kill-session -t training_main"
ssh g375 "tmux kill-session -t training_worker"
```

### ✅ Method 3: Use a Process Manager

**Using Supervisor or systemd (advanced):**

Create a systemd service that starts training and can be stopped cluster-wide. This is overkill for most use cases but useful for production.

---

## Monitoring Multi-Node Training

### Check Status of All Nodes

```bash
# For 2 nodes
./examples/qwen_image/model_training/full/check_training_status.sh 2

# For 8 nodes
./examples/qwen_image/model_training/full/check_training_status.sh 8
```

This shows:
- Which nodes are running training
- GPU utilization on each node
- Whether all nodes are active

### Manual Check (Individual Nodes)

```bash
# Check if training is running on a specific node
ssh g369 "ps aux | grep train.py"

# Check GPU usage
ssh g369 "nvidia-smi"

# Check logs (if redirected)
ssh g369 "tail -f /path/to/training.log"
```

---

## Best Practices

### 1. Always Launch in a Persistent Session

**Use tmux/screen** so you can detach and reconnect:
```bash
# Start training in tmux
tmux new -s train_g369
cd /home/kuan/workspace/repos/DiffSynth-Studio
./examples/qwen_image/model_training/full/launch_main_g369.sh

# Detach: Ctrl+B, then D
# Training continues in background

# Reconnect later
tmux attach -t train_g369
```

### 2. Redirect Output to Logs

```bash
# Modified launch with logging
./examples/qwen_image/model_training/full/launch_main_g369.sh 2>&1 | tee training_g369.log
```

### 3. Set a Timeout (Optional)

Add a timeout wrapper to auto-kill if nodes can't connect:

```bash
timeout 10m ./examples/qwen_image/model_training/full/launch_worker_g375.sh
```

If the worker can't connect to main in 10 minutes, it will auto-exit.

### 4. Use the Kill Script Before Restarting

Always clean up before starting new training:

```bash
# Clean up any stuck processes
./examples/qwen_image/model_training/full/kill_all_training.sh

# Wait a moment
sleep 5

# Verify everything is stopped
./examples/qwen_image/model_training/full/check_training_status.sh 2

# Now start fresh
./examples/qwen_image/model_training/full/launch_main_g369.sh &
./examples/qwen_image/model_training/full/launch_worker_g375.sh &
```

---

## Automated Launch + Monitoring (Advanced)

### All-in-One Launch Script

Create a master script that:
1. Kills any existing processes
2. Launches all nodes
3. Monitors status

```bash
#!/bin/bash
# master_launch.sh

NODES_MAIN="g369"
NODES_WORKERS=("g375")  # Or ("g370" "g371" ... "g376") for 8 nodes

# 1. Clean up
echo "Cleaning up existing processes..."
./examples/qwen_image/model_training/full/kill_all_training.sh
sleep 3

# 2. Launch main node in background
echo "Launching main node ${NODES_MAIN}..."
ssh ${NODES_MAIN} "cd /home/kuan/workspace/repos/DiffSynth-Studio && \
                    tmux new -d -s training_main \
                    './examples/qwen_image/model_training/full/launch_main_g369.sh'"

# Wait for main to initialize
echo "Waiting for main node to initialize..."
sleep 10

# 3. Launch worker nodes
for WORKER in "${NODES_WORKERS[@]}"; do
    echo "Launching worker node ${WORKER}..."
    ssh ${WORKER} "cd /home/kuan/workspace/repos/DiffSynth-Studio && \
                   tmux new -d -s training_worker \
                   './examples/qwen_image/model_training/full/launch_worker_g375.sh'"
    sleep 2
done

# 4. Check status
sleep 5
./examples/qwen_image/model_training/full/check_training_status.sh 2

echo ""
echo "Training launched on all nodes!"
echo "Monitor with: ./check_training_status.sh 2"
echo "Stop with: ./kill_all_training.sh"
```

---

## Troubleshooting Hung Processes

### If Processes Won't Die

```bash
# Force kill (careful!)
ssh g369 "pkill -9 -f train.py"
ssh g375 "pkill -9 -f train.py"

# Kill all Python processes (VERY aggressive - may kill other things!)
ssh g369 "pkill -9 python"
ssh g375 "pkill -9 python"
```

### If Nodes Are Stuck Waiting

Check if ports are still bound:
```bash
ssh g369 "netstat -tulpn | grep 29500"
```

If port 29500 is still in use, kill the process holding it:
```bash
ssh g369 "fuser -k 29500/tcp"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Kill all (2 nodes) | `./kill_all_training.sh` |
| Kill all (8 nodes) | `./kill_all_8nodes.sh` |
| Check status (2 nodes) | `./check_training_status.sh 2` |
| Check status (8 nodes) | `./check_training_status.sh 8` |
| Check specific node | `ssh g369 "ps aux \| grep train.py"` |
| View GPU usage | `ssh g369 nvidia-smi` |
| Force kill on node | `ssh g369 "pkill -9 -f train.py"` |

---

## Why SkyPilot Handles This Better

With SkyPilot, you get:
- **Single point of control**: `sky down qwen-8nodes` stops ALL nodes
- **Automatic cleanup**: Processes are managed centrally
- **Status monitoring**: Built-in `sky status` command
- **Log aggregation**: All logs in one place

This is one reason SkyPilot is recommended for large-scale multi-node training!

