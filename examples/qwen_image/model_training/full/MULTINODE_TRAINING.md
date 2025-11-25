# Multi-Node Training Guide

This guide covers running distributed training across multiple nodes (2 or 8 nodes).

## Quick Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Manual** | Full control, no dependencies | Manual coordination needed | Existing clusters, debugging |
| **SkyPilot** | Fully automated, handles sync | Requires SkyPilot setup | Cloud/K8s, large-scale runs |

---

## Option 1: Manual Multi-Node (Without SkyPilot)

### For 2 Nodes (g369, g375)

**On g369 (Main Node):**
```bash
cd /home/kuan/workspace/repos/DiffSynth-Studio
./examples/qwen_image/model_training/full/launch_main_g369.sh
```

**On g375 (Worker Node):**
```bash
cd /home/kuan/workspace/repos/DiffSynth-Studio
./examples/qwen_image/model_training/full/launch_worker_g375.sh
```

### For 8 Nodes

**On each node, run:**
```bash
cd /home/kuan/workspace/repos/DiffSynth-Studio
./examples/qwen_image/model_training/full/launch_multinode.sh <NODE_RANK>
```

Where `<NODE_RANK>` is:
- Node 0 (main): `./launch_multinode.sh 0`
- Node 1: `./launch_multinode.sh 1`
- Node 2: `./launch_multinode.sh 2`
- ...
- Node 7: `./launch_multinode.sh 7`

**Important:** Start node 0 first, then start other nodes within 30-60 seconds.

### Manual Launch Helper Script

Use `pdsh`, `parallel-ssh`, or this helper:

```bash
#!/bin/bash
# launch_all_nodes.sh
NODES=("g369" "g370" "g371" "g372" "g373" "g374" "g375" "g376")

for i in "${!NODES[@]}"; do
    NODE="${NODES[$i]}"
    echo "Starting node $i on ${NODE}..."
    ssh ${NODE} "cd /home/kuan/workspace/repos/DiffSynth-Studio && \
                 ./examples/qwen_image/model_training/full/launch_multinode.sh $i" &
    sleep 2  # Small delay between nodes
done

wait
echo "All nodes launched!"
```

---

## Option 2: SkyPilot (Automated - RECOMMENDED)

### Installation

```bash
pip install "skypilot[kubernetes]"  # For Kubernetes
# OR
pip install "skypilot[aws]"  # For AWS
# OR
pip install "skypilot[gcp]"  # For GCP
```

### Setup for Kubernetes (nunchaku environment)

1. **Configure SkyPilot for your K8s cluster:**
```bash
sky check kubernetes
```

2. **Verify GPU nodes are available:**
```bash
sky show-gpus --cloud kubernetes
```

### Launch Training with SkyPilot

**Single command to launch on 8 nodes:**

```bash
# From your project root
cd /home/kuan/workspace/repos/DiffSynth-Studio

# Launch the job
sky launch -c qwen-8nodes examples/qwen_image/model_training/full/skypilot_kubernetes.yaml
```

That's it! SkyPilot will:
- ✅ Provision 8 nodes with 8 GPUs each
- ✅ Sync your code to all nodes
- ✅ Set up networking between nodes
- ✅ Coordinate the distributed launch
- ✅ Stream logs from all nodes

### Monitor Training

```bash
# Check status
sky status qwen-8nodes

# Stream logs
sky logs qwen-8nodes

# SSH into main node
sky ssh qwen-8nodes

# Stop training
sky down qwen-8nodes
```

### Download Results

```bash
# Download trained model
sky rsync down qwen-8nodes:~/sky_workdir/models/train/Qwen-Image-Edit-Pico_8nodes ./models/
```

---

## Configuration Files

### 2-Node Setup (16 GPUs total)
- Config: `accelerate_config_multinode.yaml`
- Launch scripts: `launch_main_g369.sh`, `launch_worker_g375.sh`

### 8-Node Setup (64 GPUs total)
- Config: `accelerate_config_8nodes.yaml`
- Manual launch: `launch_multinode.sh`
- SkyPilot configs: `skypilot_task.yaml`, `skypilot_kubernetes.yaml`

---

## Troubleshooting

### Common Issues

**1. Connection Timeout**
- Ensure all nodes can reach the main node IP on port 29500
- Check firewall rules: `sudo ufw allow 29500/tcp`

**2. NCCL Errors**
- Set environment variables:
  ```bash
  export NCCL_DEBUG=INFO
  export NCCL_IB_DISABLE=0
  export NCCL_NET_GDR_LEVEL=2
  ```

**3. Nodes Not Joining**
- Verify main node IP is correct: `hostname -I`
- Ensure all nodes have the same code/data
- Start worker nodes within timeout window (~60s)

**4. DeepSpeed OOM**
- Increase CPU offloading in config
- Reduce `max_pixels` or batch size
- Switch to ZeRO-3 for larger models

### Debug Mode

Enable verbose logging:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Performance Tips

1. **Network**: Use InfiniBand or high-speed networking for multi-node
2. **Storage**: Use shared filesystem (NFS/Lustre) or ensure data is synced
3. **Batch Size**: Scale batch size with number of GPUs
4. **Gradient Accumulation**: Reduce if using many GPUs
5. **CPU Offload**: Consider disabling on nodes with fast GPUs to reduce overhead

---

## Why SkyPilot?

✅ **Automation**: One command to launch everything  
✅ **Portability**: Works across clouds and Kubernetes  
✅ **Cost**: Auto-stop when done  
✅ **Reproducibility**: YAML config captures everything  
✅ **Monitoring**: Built-in logging and status tracking  

**Recommended for:** Production runs, cloud environments, teams, reproducible research

