# Quick Start: 8-Node Training

## TL;DR - Three Ways to Launch

### 🚀 Method 1: SkyPilot (RECOMMENDED - Fully Automated)

```bash
# Install SkyPilot
pip install "skypilot[kubernetes]"

# Launch on 8 nodes with one command
sky launch -c qwen-8nodes examples/qwen_image/model_training/full/skypilot_kubernetes.yaml

# Monitor
sky logs qwen-8nodes --follow

# Download results when done
sky rsync down qwen-8nodes:~/sky_workdir/models/train/Qwen-Image-Edit-Pico_8nodes ./models/
```

**Pros:** Fully automated, handles everything  
**Cons:** Requires SkyPilot setup  
**Time to launch:** ~5 minutes (including cluster provisioning)

---

### ⚡ Method 2: Automated SSH (Semi-Automated)

```bash
# Just run this from your main machine
./examples/qwen_image/model_training/full/launch_all_8nodes.sh
```

**Pros:** Simple, one command, no dependencies  
**Cons:** Requires SSH access to all nodes, manual node list  
**Time to launch:** ~1 minute

**Before running:** Update node list in `launch_all_8nodes.sh` if your nodes aren't g369-g376

---

### 🔧 Method 3: Manual (Full Control)

**On each node, SSH and run:**
```bash
# Node 0 (main):
ssh g369 "cd /home/kuan/workspace/repos/DiffSynth-Studio && ./examples/qwen_image/model_training/full/launch_multinode.sh 0"

# Node 1:
ssh g370 "cd /home/kuan/workspace/repos/DiffSynth-Studio && ./examples/qwen_image/model_training/full/launch_multinode.sh 1"

# Node 2:
ssh g371 "cd /home/kuan/workspace/repos/DiffSynth-Studio && ./examples/qwen_image/model_training/full/launch_multinode.sh 2"

# ... and so on for nodes 3-7
```

**Pros:** Maximum control, good for debugging  
**Cons:** Manual, tedious for many nodes  
**Time to launch:** ~5-10 minutes

---

## Configuration Summary

| Setup | Nodes | GPUs | Total GPUs | Config File |
|-------|-------|------|------------|-------------|
| 2-node | 2 | 8/node | 16 | `accelerate_config_multinode.yaml` |
| 8-node | 8 | 8/node | 64 | `accelerate_config_8nodes.yaml` |

---

## Pre-Launch Checklist

### For All Methods:
- [ ] All nodes have the same code version
- [ ] All nodes can access the data path
- [ ] Network connectivity between nodes (port 29500)
- [ ] Main node IP is correct (currently: `10.15.38.17`)

### For Manual/SSH Methods Only:
- [ ] SSH keys set up for passwordless login to all nodes
- [ ] Node names in scripts match your actual nodes
- [ ] Same Python environment on all nodes

### For SkyPilot Only:
- [ ] SkyPilot installed: `pip install skypilot[kubernetes]`
- [ ] Kubernetes configured: `sky check kubernetes`
- [ ] Enough GPU quota available

---

## Verify Training is Running

```bash
# Check GPU usage on any node
nvidia-smi

# Check training logs (main node)
tail -f <path_to_logs>

# Check if all processes are communicating
ps aux | grep "accelerate"
```

---

## Quick Troubleshooting

**Problem:** Nodes can't connect  
**Solution:** Check firewall: `sudo ufw allow 29500/tcp`

**Problem:** Different node has different data  
**Solution:** Use shared NFS mount or rsync data to all nodes

**Problem:** SkyPilot can't find nodes  
**Solution:** Run `sky check kubernetes` and ensure cluster is configured

**Problem:** Out of memory  
**Solution:** Increase CPU offload or reduce `max_pixels` in training script

---

## Need Help?

See detailed guide: `MULTINODE_TRAINING.md`

