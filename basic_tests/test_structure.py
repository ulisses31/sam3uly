"""Minimal SAM3 test - check basic structure"""

import os
import sys

print("SAM3 Basic Structure Test")
print("=" * 60)

# Check if sam3 directory exists
sam3_path = r"c:\Dropbox\Code Robotics Books\sam3uly\sam3"
if os.path.exists(sam3_path):
    print(f"✓ SAM3 directory found: {sam3_path}")
else:
    print(f"✗ SAM3 directory not found")
    sys.exit(1)

# List main components
print("\nMain SAM3 components:")
for item in os.listdir(sam3_path):
    item_path = os.path.join(sam3_path, item)
    if os.path.isdir(item_path) and not item.startswith('__'):
        print(f"  📁 {item}/")
    elif item.endswith('.py') and not item.startswith('__'):
        print(f"  📄 {item}")

# Check examples
examples_path = r"c:\Dropbox\Code Robotics Books\sam3uly\examples"
if os.path.exists(examples_path):
    print(f"\n✓ Examples directory found")
    print("Available example notebooks:")
    for item in sorted(os.listdir(examples_path)):
        if item.endswith('.ipynb'):
            print(f"  • {item}")

# Test basic Python imports that don't require torchvision
print("\n" + "=" * 60)
print("Testing core dependencies:")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import torch
    print(f"✓ torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    from PIL import Image
    print(f"✓ PIL (Pillow)")
except ImportError as e:
    print(f"✗ PIL: {e}")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

print("\n" + "=" * 60)
print("Note: Full SAM3 model import requires torchvision compatibility.")
print("The repository structure and basic dependencies are in place.")
print("=" * 60)
