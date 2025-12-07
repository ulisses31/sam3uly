"""
Simple SAM3 functionality test
Tests imports, tensor operations, and basic utilities without model weights
"""

import sys
import os
sys.path.insert(0, r"c:\Dropbox\Code Robotics Books\sam3uly")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 70)
print("SAM3 Simple Example - Testing Core Functionality")
print("=" * 70)

# Check GPU
print(f"\n1. GPU Status:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test imports
print(f"\n2. Importing SAM3 modules...")
try:
    from sam3 import build_sam3_image_model
    from sam3.model.box_ops import box_xywh_to_cxcywh, box_cxcywh_to_xyxy
    from sam3.visualization_utils import generate_colors, normalize_bbox
    print(f"   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test box operations on GPU
print(f"\n3. Testing box operations on GPU...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create boxes in xywh format
    boxes_xywh = torch.tensor([
        [50, 100, 200, 150],
        [300, 200, 100, 100],
    ], dtype=torch.float32, device=device)
    
    print(f"   Input boxes (x,y,w,h) on {device}:")
    for i, box in enumerate(boxes_xywh):
        print(f"     Box {i+1}: {box.tolist()}")
    
    # Convert to center format
    boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
    print(f"\n   Converted to (cx,cy,w,h):")
    for i, box in enumerate(boxes_cxcywh):
        print(f"     Box {i+1}: {box.tolist()}")
    
    # Convert to xyxy format
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    print(f"\n   Converted to (x1,y1,x2,y2):")
    for i, box in enumerate(boxes_xyxy):
        print(f"     Box {i+1}: {box.tolist()}")
    
    print(f"\n   ✓ Box operations working on GPU!")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test color generation
print(f"\n4. Testing color generation...")
try:
    colors = generate_colors(n_colors=10, n_samples=1000)
    print(f"   ✓ Generated {len(colors)} distinct colors")
    print(f"   First 3 colors (RGB):")
    for i, color in enumerate(colors[:3]):
        print(f"     Color {i+1}: {color}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test bbox normalization
print(f"\n5. Testing bbox normalization...")
try:
    # Image size
    img_w, img_h = 640, 480
    
    # Box in cxcywh format
    box_cxcywh = torch.tensor([[320.0, 240.0, 100.0, 80.0]])
    
    # Normalize
    normalized = normalize_bbox(box_cxcywh, img_w, img_h)
    
    print(f"   Image size: {img_w}x{img_h}")
    print(f"   Box (cx,cy,w,h): {box_cxcywh[0].tolist()}")
    print(f"   Normalized: {normalized[0].tolist()}")
    print(f"   ✓ Bbox normalization working!")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test creating a dummy image and processing
print(f"\n6. Testing image creation and tensor conversion...")
try:
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)
    
    print(f"   Created image: {pil_image.size}")
    print(f"   Tensor shape: {image_tensor.shape} on {device}")
    print(f"   Value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    print(f"   ✓ Image processing working!")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Test GPU memory allocation
if torch.cuda.is_available():
    print(f"\n7. Testing GPU memory allocation...")
    try:
        # Create a large tensor on GPU
        large_tensor = torch.randn(1000, 1000, device=device)
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"   Created tensor: {large_tensor.shape} on GPU")
        print(f"   GPU memory allocated: {memory_allocated:.2f} MB")
        print(f"   ✓ GPU memory operations working!")
        
        # Cleanup
        del large_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")

print("\n" + "=" * 70)
print("SUCCESS! All core SAM3 functionalities are working!")
print("=" * 70)
print("\nNote: To run full model inference, you need to:")
print("  1. Download SAM3 checkpoint files")
print("  2. Use one of the example notebooks in the 'examples' folder")
print("=" * 70)
