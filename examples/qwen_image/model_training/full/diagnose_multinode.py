#!/usr/bin/env python3
"""
Minimal test to diagnose multi-node distributed setup
Run this with accelerate on each node simultaneously to test coordination
"""

import os
import socket
import torch
import torch.distributed as dist
from accelerate import Accelerator

print(f"\n{'='*60}")
print(f"Multi-Node Diagnostic Test")
print(f"{'='*60}")

# Print environment variables
print(f"\nEnvironment Variables:")
for key in ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'NODE_RANK']:
    print(f"  {key}: {os.environ.get(key, 'NOT SET')}")

print(f"\nHostname: {socket.gethostname()}")
print(f"IP: {socket.gethostbyname(socket.gethostname())}")

# Try to initialize accelerator
print(f"\nInitializing Accelerator...")
try:
    accelerator = Accelerator()
    
    print(f"\n{'='*60}")
    print(f"Accelerator Info:")
    print(f"  Process index: {accelerator.process_index}")
    print(f"  Local process index: {accelerator.local_process_index}")
    print(f"  Num processes: {accelerator.num_processes}")
    print(f"  Is main process: {accelerator.is_main_process}")
    print(f"  Device: {accelerator.device}")
    print(f"  Distributed type: {accelerator.distributed_type}")
    print(f"{'='*60}")
    
    # Test communication
    if accelerator.num_processes > 1:
        test_tensor = torch.tensor([float(accelerator.process_index)]).to(accelerator.device)
        print(f"\nBefore all_reduce: {test_tensor.item()}")
        
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        expected = sum(range(accelerator.num_processes))
        actual = test_tensor.item()
        
        print(f"After all_reduce: {actual}")
        print(f"Expected sum: {expected}")
        
        if abs(actual - expected) < 0.1:
            print(f"✅ All {accelerator.num_processes} processes are communicating!")
        else:
            print(f"❌ Communication FAILED! Only some processes connected.")
    else:
        print(f"\n⚠️  Only 1 process detected - no distributed setup!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*60}\n")

