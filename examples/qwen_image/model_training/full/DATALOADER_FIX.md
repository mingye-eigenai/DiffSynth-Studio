# DataLoader Distribution Issue - Why Steps Don't Decrease with More GPUs

## The Problem

You're seeing **32,217 steps per epoch** on both:
- 1 node (8 GPUs) 
- 2 nodes (16 GPUs)

**Expected behavior:**
- 1 node: 32,217 / 8 = ~4,027 steps
- 2 nodes: 32,217 / 16 = ~2,014 steps

## Root Cause

The training script creates a DataLoader with:
```python
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
```

Issues:
1. **No explicit batch_size** → defaults to `batch_size=1`
2. **shuffle=True with DeepSpeed** → may not auto-inject DistributedSampler
3. **Each GPU processes the entire dataset** instead of a partition

## Diagnosis

Run this on your training node:

```bash
cd /home/kuan/workspace/repos/DiffSynth-Studio
accelerate launch \
  --config_file examples/qwen_image/model_training/full/accelerate_config_multinode.yaml \
  examples/qwen_image/model_training/full/verify_distributed_setup.py
```

Look for the output:
- ✅ **SUCCESS**: Data is properly distributed
- ❌ **ERROR**: Data is NOT being distributed (this is your issue)

## Solutions

### Solution 1: Modify DeepSpeed Config (Recommended)

Add `"train_batch_size": "auto"` to your DeepSpeed config:

```yaml
deepspeed_config:
  gradient_accumulation_steps: 1
  train_batch_size: "auto"  # Add this
  train_micro_batch_size_per_gpu: 1  # Add this
  offload_optimizer_device: 'cpu'
  offload_param_device: 'cpu'
  zero3_init_flag: false
  zero_stage: 2
```

This tells DeepSpeed:
- `train_micro_batch_size_per_gpu: 1` → each GPU processes 1 sample per step
- `train_batch_size: auto` → auto-calculated as `num_gpus * micro_batch_size * gradient_accum`
- Data will be properly distributed across processes

### Solution 2: Patch the Training Code

If Solution 1 doesn't work, you need to modify `diffsynth/trainers/utils.py`:

**Current code (line 545):**
```python
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, ...)
model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
```

**Fixed code:**
```python
# Don't use shuffle=True directly - let Accelerate handle it
dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    dispatch_batches=False,  # Let DeepSpeed handle batching
    kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
)

# Prepare with explicit data splitting
model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

# Manually set shuffling via DistributedSampler
if accelerator.distributed_type == accelerate.DistributedType.DEEPSPEED:
    dataloader.sampler.set_epoch(0)  # Enable shuffling
```

### Solution 3: Check Actual Batch Processing

Add logging to verify data distribution. Modify your training to add these prints:

```python
# Add to training loop
if accelerator.is_main_process:
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Expected if distributed: {len(dataset) // accelerator.num_processes}")
```

## Recommended Action

1. **First**, try Solution 1 (modify DeepSpeed config)
2. **Run** the verification script to confirm
3. **If still broken**, contact the library maintainers or modify the core training code

## Why This Matters

If data is not distributed:
- **Wasted computation**: All GPUs train on the same data (redundant)
- **No speedup**: Training takes the same time regardless of GPUs
- **Wrong gradient averaging**: Gradients are averaged across duplicate data

This explains why your 2-node training isn't faster!

