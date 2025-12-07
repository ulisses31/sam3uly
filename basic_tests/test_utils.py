"""Test SAM3 utilities without full model loading"""

import sys
import os

# Add sam3 to path
sys.path.insert(0, r"c:\Dropbox\Code Robotics Books\sam3uly")

print("Testing SAM3 Utilities")
print("=" * 60)

# Test 1: Import visualization utils (doesn't need torchvision)
try:
    from sam3.visualization_utils import normalize_bbox
    print("✓ Successfully imported visualization_utils")
    
    # Test normalize_bbox function
    bbox = [10, 20, 100, 200]  # x, y, w, h
    image_shape = (480, 640)  # height, width
    normalized = normalize_bbox(bbox, image_shape)
    print(f"  Test: normalize_bbox({bbox}, {image_shape})")
    print(f"  Result: {normalized}")
    
except Exception as e:
    print(f"✗ visualization_utils test failed: {e}")

# Test 2: Import logger
try:
    from sam3.logger import setup_logger
    print("✓ Successfully imported logger")
except Exception as e:
    print(f"✗ logger import failed: {e}")

# Test 3: Check box operations (might work without full model)
try:
    from sam3.model.box_ops import box_xywh_to_cxcywh
    import torch
    
    print("✓ Successfully imported box_ops")
    
    # Test box conversion
    boxes = torch.tensor([[10, 20, 100, 200], [50, 60, 150, 250]], dtype=torch.float32)
    converted = box_xywh_to_cxcywh(boxes)
    print(f"  Test: box_xywh_to_cxcywh")
    print(f"  Input:  {boxes[0].tolist()}")
    print(f"  Output: {converted[0].tolist()}")
    
except Exception as e:
    print(f"✗ box_ops test failed: {e}")

print("=" * 60)
print("Basic utility tests completed!")
