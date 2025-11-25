#!/usr/bin/env python3
"""
Quick script to verify distributed data loading is working correctly.
Run this with your accelerate config to see if data is being split properly.
"""

import torch
from accelerate import Accelerator
from diffsynth.trainers.unified_dataset import UnifiedDataset
import os

# Create a simple dataset
dataset = UnifiedDataset(
    base_path="data/",
    metadata_path="/home/kuan/workspace/repos/DiffSynth-Studio/data/pico-banana-400k/openimages/metadata_sft_clean.csv",
    repeat=1,
    data_file_keys=["image", "edit_image"],
    main_data_operator=UnifiedDataset.default_image_operator(
        base_path="data/",
        max_pixels=1048576,
    )
)

print(f"Total dataset size: {len(dataset)}")

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    shuffle=True, 
    collate_fn=lambda x: x[0], 
    num_workers=2
)

print(f"Dataloader length BEFORE accelerator.prepare: {len(dataloader)}")

# Initialize accelerator
accelerator = Accelerator()

# Prepare dataloader
dataloader = accelerator.prepare(dataloader)

print(f"\n=== Distributed Setup ===")
print(f"Process index: {accelerator.process_index}")
print(f"Num processes: {accelerator.num_processes}")
print(f"Is main process: {accelerator.is_main_process}")
print(f"Dataloader length AFTER accelerator.prepare: {len(dataloader)}")
print(f"Expected steps per epoch: {len(dataset)} / {accelerator.num_processes} = {len(dataset) // accelerator.num_processes}")
print(f"Actual steps per epoch: {len(dataloader)}")

# Check if data is properly distributed
if len(dataloader) == len(dataset):
    print(f"\n❌ ERROR: Data is NOT being distributed! Each GPU will process the full dataset.")
    print(f"   This will cause {accelerator.num_processes}x redundant computation!")
elif len(dataloader) == len(dataset) // accelerator.num_processes:
    print(f"\n✅ SUCCESS: Data is properly distributed across {accelerator.num_processes} processes")
else:
    print(f"\n⚠️  WARNING: Unexpected dataloader length. Manual investigation needed.")

