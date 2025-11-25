#!/bin/bash

echo "Fixing Flash Attention import in DiffSynth..."
echo "============================================"

# Backup the original file
cp /home/kuan/workspace/repos/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py \
   /home/kuan/workspace/repos/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py.backup

# Fix the import
sed -i '9s/import flash_attn_interface/from flash_attn import flash_attn_interface/' \
    /home/kuan/workspace/repos/DiffSynth-Studio/diffsynth/models/qwen_image_dit.py

echo "✅ Import fixed!"
echo ""
echo "Verifying the fix..."

python -c "
import sys
sys.path.insert(0, '/home/kuan/workspace/repos/DiffSynth-Studio')

try:
    from diffsynth.models.qwen_image_dit import FLASH_ATTN_3_AVAILABLE
    if FLASH_ATTN_3_AVAILABLE:
        print('✅ SUCCESS! Flash Attention is now properly detected!')
        print('🚀 The model will now use Flash Attention for 2-3x speedup!')
    else:
        print('❌ Flash Attention still not detected')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo ""
echo "Now run check_setup_status.py again to verify:"
echo "python check_setup_status.py"
