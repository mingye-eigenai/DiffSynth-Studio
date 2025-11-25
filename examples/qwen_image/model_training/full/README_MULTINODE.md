# Complete Multi-Node Training Guide

This directory contains everything you need to run distributed training across multiple nodes.

## 🚀 Quick Start (2 Nodes - g369 & g375)

### Automated Launch (Recommended)
```bash
./examples/qwen_image/model_training/full/AUTO_LAUNCH_2NODES.sh
```

This will:
- ✅ Kill any existing training processes
- ✅ Launch main node (g369)
- ✅ Launch worker node (g375)
- ✅ Verify both are running
- ✅ Create log files with timestamps

### Manual Launch
```bash
# On g369:
./examples/qwen_image/model_training/full/launch_main_g369_FIXED.sh

# On g375 (within 30 seconds):
./examples/qwen_image/model_training/full/launch_worker_g375_FIXED.sh
```

### Stop Training
```bash
./examples/qwen_image/model_training/full/kill_all_training.sh
```

⚠️ **Important**: You MUST use the kill script to stop all nodes. Killing just one node leaves others hanging!

---

## 📊 Monitoring

### Check Training Status
```bash
# For 2 nodes
./examples/qwen_image/model_training/full/check_training_status.sh 2

# For 8 nodes
./examples/qwen_image/model_training/full/check_training_status.sh 8
```

### View Logs
```bash
# If using AUTO_LAUNCH script
tail -f logs/multinode_training_*/main_g369.log
tail -f logs/multinode_training_*/worker_g375.log

# If running manually
# Logs will be on each node's terminal/tmux session
```

---

## 📁 File Reference

### Configuration Files

| File | Purpose | Nodes | GPUs | Status |
|------|---------|-------|------|--------|
| `accelerate_config_multinode_FIXED.yaml` | 2-node config with proper data distribution | 2 | 16 | ✅ Use this |
| `accelerate_config_8nodes_FIXED.yaml` | 8-node config with proper data distribution | 8 | 64 | ✅ Use this |
| `accelerate_config_multinode.yaml` | Original 2-node config | 2 | 16 | ❌ Has bug (no data distribution) |
| `accelerate_config_8nodes.yaml` | Original 8-node config | 8 | 64 | ❌ Has bug (no data distribution) |

### Launch Scripts

| Script | Purpose |
|--------|---------|
| `AUTO_LAUNCH_2NODES.sh` | Automated 2-node launcher with cleanup & logging |
| `launch_main_g369_FIXED.sh` | Main node launcher (2 nodes, FIXED) |
| `launch_worker_g375_FIXED.sh` | Worker node launcher (2 nodes, FIXED) |
| `launch_multinode.sh` | Universal launcher (pass node rank as arg) |
| `launch_all_8nodes.sh` | SSH launcher for all 8 nodes |

### Management Scripts

| Script | Purpose |
|--------|---------|
| `kill_all_training.sh` | Stop training on 2 nodes |
| `kill_all_8nodes.sh` | Stop training on 8 nodes |
| `check_training_status.sh` | Check which nodes are running |
| `verify_distributed_setup.py` | Diagnostic: verify data distribution works |

### Documentation

| File | Content |
|------|---------|
| `README_MULTINODE.md` | This file - overview of everything |
| `FIX_SUMMARY.md` | Explanation of the data distribution bug & fix |
| `DATALOADER_FIX.md` | Technical details on the dataloader issue |
| `MULTINODE_MANAGEMENT.md` | How to manage multi-node jobs (kill, monitor, etc.) |
| `MULTINODE_TRAINING.md` | General multi-node training guide |
| `QUICK_START_8NODES.md` | Quick reference for 8-node training |

### SkyPilot Configs

| File | Purpose |
|------|---------|
| `skypilot_task.yaml` | General cloud/K8s launcher |
| `skypilot_kubernetes.yaml` | Optimized for Kubernetes (nunchaku) |

---

## 🐛 Common Issues & Solutions

### Issue 1: Same Steps Per Epoch on 1-Node vs 2-Node

**Problem:** Both show 32,217 steps instead of halving with more GPUs

**Solution:** Use the `_FIXED` configs that include:
```yaml
train_batch_size: "auto"
train_micro_batch_size_per_gpu: 1
```

**Read:** `FIX_SUMMARY.md`

### Issue 2: Killing One Node Doesn't Stop Others

**Problem:** When you kill g369, g375 keeps running (and vice versa)

**Why:** Each node is an independent process. They don't auto-kill each other.

**Solution:** Use the kill script:
```bash
./examples/qwen_image/model_training/full/kill_all_training.sh
```

**Read:** `MULTINODE_MANAGEMENT.md`

### Issue 3: Worker Node Can't Connect to Main

**Symptoms:**
- Worker hangs at "Waiting for other processes..."
- Timeout errors

**Checklist:**
- [ ] Main node started FIRST?
- [ ] Worker started within 60 seconds of main?
- [ ] Port 29500 open? (`sudo ufw allow 29500/tcp`)
- [ ] Main IP correct in config? (Currently: `10.15.38.17`)
- [ ] Both nodes can ping each other?

**Debug:**
```bash
# On worker node, check if main is reachable
telnet 10.15.38.17 29500

# On main node, check if port is listening
netstat -tulpn | grep 29500
```

### Issue 4: Out of Memory

**Solutions:**
1. Increase gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 8  # Instead of 1
   ```
2. Enable more aggressive CPU offloading
3. Reduce `max_pixels` or image resolution
4. Switch to ZeRO-3 for larger models

---

## 📈 Expected Performance

### With FIXED Configs

| Setup | Nodes | GPUs | Steps/Epoch | Expected Speedup |
|-------|-------|------|-------------|------------------|
| Single node | 1 | 8 | ~4,027 | 1x (baseline) |
| Two nodes | 2 | 16 | ~2,014 | ~2x |
| Eight nodes | 8 | 64 | ~503 | ~8x |

Total dataset: 32,217 samples

### Training Time Estimates

Assuming ~2 seconds per step (varies by GPU):

| Setup | Steps/Epoch | Time/Epoch | Total (5 epochs) |
|-------|-------------|------------|------------------|
| 1 node | 4,027 | ~2.2 hours | ~11 hours |
| 2 nodes | 2,014 | ~1.1 hours | ~5.5 hours |
| 8 nodes | 503 | ~17 minutes | ~1.4 hours |

---

## 🎯 Best Practices

### 1. Always Use Kill Script Before Restarting

```bash
# Clean slate
./examples/qwen_image/model_training/full/kill_all_training.sh

# Verify
./examples/qwen_image/model_training/full/check_training_status.sh 2

# Then launch
./examples/qwen_image/model_training/full/AUTO_LAUNCH_2NODES.sh
```

### 2. Use tmux/screen for Manual Launch

```bash
# On g369
tmux new -s main
./examples/qwen_image/model_training/full/launch_main_g369_FIXED.sh
# Ctrl+B, D to detach

# On g375
tmux new -s worker
./examples/qwen_image/model_training/full/launch_worker_g375_FIXED.sh
# Ctrl+B, D to detach
```

### 3. Monitor Regularly

```bash
# Set up a watch to auto-refresh status
watch -n 10 './examples/qwen_image/model_training/full/check_training_status.sh 2'
```

### 4. For Production: Use SkyPilot

For serious multi-node training, especially 8+ nodes:

```bash
sky launch -c qwen-8nodes examples/qwen_image/model_training/full/skypilot_kubernetes.yaml
```

**Benefits:**
- One command to launch everything
- Automatic cleanup on stop
- Centralized logging
- Works across clouds/K8s

---

## 🔧 Advanced Usage

### 8-Node Training

**Automated (SSH):**
```bash
./examples/qwen_image/model_training/full/launch_all_8nodes.sh
```

**SkyPilot (Recommended):**
```bash
sky launch -c qwen-8nodes examples/qwen_image/model_training/full/skypilot_kubernetes.yaml
```

**Manual:**
```bash
# On each node
./examples/qwen_image/model_training/full/launch_multinode.sh <RANK>

# Where RANK is:
# g369: 0
# g370: 1
# g371: 2
# ... etc
```

**Stop:**
```bash
./examples/qwen_image/model_training/full/kill_all_8nodes.sh
```

---

## 🆘 Getting Help

1. **Check logs** in `logs/multinode_training_*/`
2. **Run verification**: `verify_distributed_setup.py`
3. **Check status**: `check_training_status.sh`
4. **Read docs** in this directory (especially `MULTINODE_MANAGEMENT.md`)

---

## 📚 Related Documentation

- **Data Distribution Fix**: `FIX_SUMMARY.md` - Why steps weren't halving
- **Process Management**: `MULTINODE_MANAGEMENT.md` - How to kill/monitor nodes
- **Technical Details**: `DATALOADER_FIX.md` - Deep dive on the bug
- **8-Node Quick Start**: `QUICK_START_8NODES.md` - Fast reference for 8 nodes
- **General Guide**: `MULTINODE_TRAINING.md` - Comprehensive multi-node guide

---

## ✅ Checklist Before Training

- [ ] Read `FIX_SUMMARY.md` to understand data distribution fix
- [ ] Clean up existing processes: `./kill_all_training.sh`
- [ ] Verify configs use `_FIXED` versions
- [ ] Ensure port 29500 is open on all nodes
- [ ] Check main node IP is correct (10.15.38.17)
- [ ] Both nodes have same code version
- [ ] Both nodes have access to same data paths
- [ ] Launch main node FIRST
- [ ] Launch worker(s) within 60 seconds
- [ ] Monitor with `check_training_status.sh`

Happy training! 🚀

