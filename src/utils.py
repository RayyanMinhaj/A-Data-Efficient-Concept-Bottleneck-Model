import torch
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import os

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_device():
    return DEVICE





# Loads both CLIP and SAM
def load_models(sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b"):

    print(f"Loading models on {DEVICE}...")
    
    if not os.path.exists(sam_checkpoint):
        raise FileNotFoundError(f"SAM Checkpoint not found at: {sam_checkpoint}. Please download it.")

    # 1. Load CLIP
    # Returns (model, preprocess)
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=DEVICE)
    clip_model.eval()

    # 2. Load SAM
    print("Loading SAM...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    
    # 3. Create Mask Generator
    # This should return a SamAutomaticMaskGenerator object
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Models loaded successfully.")
    return clip_model, clip_preprocess, mask_generator





def crop_image_from_mask(image_pil, mask_dict, expansion=0):

    x, y, w, h = mask_dict['bbox']
    
    width, height = image_pil.size
    x = max(0, x - expansion)
    y = max(0, y - expansion)
    w = min(width - x, w + (2 * expansion))
    h = min(height - y, h + (2 * expansion))
    
    cropped = image_pil.crop((x, y, x+w, y+h))
    return cropped







def get_image_paths(dataset_root, n_images_per_class=50):
    import os
    import random
    
    image_paths = []
    if not os.path.exists(dataset_root):
        print(f"ERROR: Dataset root {dataset_root} does not exist!")
        return []

    classes = os.listdir(dataset_root)
    
    for class_name in classes:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        all_imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.shuffle(all_imgs)
        selected = all_imgs[:n_images_per_class]
        image_paths.extend(selected)
        
    return image_paths
