"""Simple test to verify SAM3 installation and basic imports"""

print("Testing SAM3 installation...")
print("-" * 50)

# Test basic Python imports
print("✓ Python running")

# Test numpy
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy not available: {e}")

# Test torch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch not available: {e}")

# Test sam3 package
try:
    import sam3
    print(f"✓ SAM3 package imported")
    print(f"  - SAM3 location: {sam3.__file__}")
except ImportError as e:
    print(f"✗ SAM3 not available: {e}")

# Test sam3 model builder
try:
    from sam3 import build_sam3_image_model
    print(f"✓ SAM3 model builder available")
except ImportError as e:
    print(f"✗ SAM3 model builder not available: {e}")

print("-" * 50)
print("Test complete!")
