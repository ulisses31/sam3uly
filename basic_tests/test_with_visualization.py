"""
SAM3 Inference Example with Proper Visualization
Using downloaded model weights to perform segmentation and save results
"""

import sys
import os
sys.path.insert(0, r"c:\Dropbox\Code Robotics Books\sam3uly")

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("=" * 70)
print("SAM3 INFERENCE WITH VISUALIZATION")
print("=" * 70)

# Setup
print("\n1. Setting up environment...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Import SAM3
print("\n2. Importing SAM3 modules...")
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, generate_colors

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Load model
print("\n3. Loading SAM3 model...")
bpe_path = r"C:\Dropbox\Code Robotics Books\sam3uly\assets\bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
print(f"   ✓ Model loaded on {next(model.parameters()).device}")

# Load test image
print("\n4. Loading test image...")
image_path = r"C:\Dropbox\Code Robotics Books\sam3uly\assets\images\test_image.jpg"
image = Image.open(image_path)
width, height = image.size
print(f"   Size: {width}x{height}")

# Initialize processor
print("\n5. Initializing processor...")
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

# Generate colors for visualization
colors = generate_colors(n_colors=50)

def save_results(img, results, output_path, title):
    """Save segmentation results as an image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    nb_objects = len(results["scores"])
    print(f"   Found {nb_objects} object(s)")
    
    for i in range(nb_objects):
        color = colors[i % len(colors)]
        
        # Plot mask
        mask = results["masks"][i].squeeze(0).cpu().numpy()
        mask_overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        mask_overlay[:, :, :3] = color
        mask_overlay[:, :, 3] = mask * 0.5  # Alpha channel
        ax.imshow(mask_overlay)
        
        # Plot bounding box
        box = results["boxes"][i].cpu().numpy()  # XYXY format
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add text label
        prob = results["scores"][i].item()
        ax.text(
            x1, y1 - 5,
            f"ID={i}, conf={prob:.2f}",
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"   ✓ Saved: {os.path.basename(output_path)}")

# Test 1: Text prompt - "shoe"
print("\n6. Test 1: Text prompt 'shoe'...")
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="shoe")

output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\result_shoe.png"
save_results(image.copy(), inference_state, output_path, "SAM3: Text Prompt 'shoe'")

# Test 2: Text prompt - "person"
print("\n7. Test 2: Text prompt 'person'...")
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="person")

output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\result_person.png"
save_results(image.copy(), inference_state, output_path, "SAM3: Text Prompt 'person'")

# Test 3: Box prompt
print("\n8. Test 3: Box prompt...")
box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()

processor.reset_all_prompts(inference_state)
inference_state = processor.add_geometric_prompt(
    state=inference_state, box=norm_box_cxcywh, label=True
)

output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\result_box.png"
save_results(image.copy(), inference_state, output_path, "SAM3: Box Prompt")

# Test 4: Different text prompt - "table"
print("\n9. Test 4: Text prompt 'table'...")
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="table")

output_path = r"C:\Dropbox\Code Robotics Books\sam3uly\basic_tests\result_table.png"
save_results(image.copy(), inference_state, output_path, "SAM3: Text Prompt 'table'")

print("\n" + "=" * 70)
print("SUCCESS! All tests completed!")
print("=" * 70)
print("\nGenerated images in basic_tests/:")
print("  - result_shoe.png")
print("  - result_person.png")
print("  - result_box.png")
print("  - result_table.png")
print("=" * 70)
