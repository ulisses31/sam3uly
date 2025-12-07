"""
SAM3 Real Inference Example
Using downloaded model weights to perform actual segmentation
"""

import sys
import os
sys.path.insert(0, r"c:\Dropbox\Code Robotics Books\sam3uly")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 70)
print("SAM3 REAL INFERENCE EXAMPLE")
print("=" * 70)

# Setup
print("\n1. Setting up environment...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Import SAM3
print("\n2. Importing SAM3 modules...")
try:
    from sam3 import build_sam3_image_model
    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import normalize_bbox, plot_results
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Enable GPU optimizations
print("\n3. Configuring GPU optimizations...")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    print("   ✓ Enabled TF32 and bfloat16")

# Load model
print("\n4. Loading SAM3 model...")
try:
    # Try HuggingFace cache first
    model_path = r"C:\Users\pinto\.cache\huggingface\hub\models--facebook--sam3"
    bpe_path = r"C:\Dropbox\Code Robotics Books\sam3uly\assets\bpe_simple_vocab_16e6.txt.gz"
    
    print(f"   BPE vocab: {bpe_path}")
    print(f"   Loading model from HuggingFace cache...")
    
    model = build_sam3_image_model(bpe_path=bpe_path)
    print(f"   ✓ Model loaded successfully!")
    print(f"   Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    print("\n   Trying alternative model path...")
    try:
        checkpoint_path = r"C:\Dropbox\Code Robotics Books\sam3-models-weights\sam3.pt"
        print(f"   Loading from: {checkpoint_path}")
        
        model = build_sam3_image_model(bpe_path=bpe_path, checkpoint=checkpoint_path)
        print(f"   ✓ Model loaded from checkpoint!")
    except Exception as e2:
        print(f"   ✗ Failed: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Load test image
print("\n5. Loading test image...")
try:
    image_path = r"C:\Dropbox\Code Robotics Books\sam3uly\assets\images\test_image.jpg"
    image = Image.open(image_path)
    width, height = image.size
    print(f"   Image: {image_path}")
    print(f"   Size: {width}x{height}")
    print(f"   ✓ Image loaded")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Initialize processor
print("\n6. Initializing SAM3 processor...")
try:
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    print(f"   ✓ Processor initialized")
    print(f"   Confidence threshold: 0.5")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Text prompt
print("\n7. Testing text prompt: 'shoe'...")
try:
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt="shoe")
    
    # Get results
    masks = inference_state.get('masks', None)
    if masks is not None:
        print(f"   ✓ Segmentation complete!")
        print(f"   Found {len(masks)} mask(s)")
    else:
        print(f"   No masks found")
    
    # Save result
    output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\output_text_prompt.png"
    fig = plot_results(image.copy(), inference_state)
    if fig is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Result saved to: output_text_prompt.png")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Box prompt
print("\n8. Testing box prompt...")
try:
    # Define a box around an object (x, y, width, height)
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    
    print(f"   Box (x,y,w,h): {box_input_xywh.flatten().tolist()}")
    print(f"   Normalized: {norm_box_cxcywh}")
    
    processor.reset_all_prompts(inference_state)
    inference_state = processor.add_geometric_prompt(
        state=inference_state, box=norm_box_cxcywh, label=True
    )
    
    masks = inference_state.get('masks', None)
    if masks is not None:
        print(f"   ✓ Segmentation complete!")
        print(f"   Found {len(masks)} mask(s)")
    
    # Save result
    output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\output_box_prompt.png"
    fig = plot_results(image.copy(), inference_state)
    if fig is not None:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"   ✓ Result saved to: output_box_prompt.png")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("INFERENCE COMPLETE!")
print("=" * 70)
print("\nCheck the basic_tests folder for output images:")
print("  - output_text_prompt.png")
print("  - output_box_prompt.png")
print("=" * 70)
