import torch
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_device():
    return DEVICE


# Loads both CLIP and SAM
def load_models(sam_checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b"):
    
    print(f"Loading models on {DEVICE}...")
    
    # Load CLIP
    # ViT-B/16 is a standard, good balance of speed/performance
    clip_model, clip_preprocess = clip.load("ViT-B/16", device=DEVICE)
    clip_model.eval()

    # Load SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)
    
    # this generator runs over the whole image and finds all masks
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Models loaded successfully.")
    return clip_model, clip_preprocess, mask_generator






def crop_image_from_mask(image_pil, mask_dict, expansion=0):

    x, y, w, h = mask_dict['bbox']
    
    # Optional: Expand box slightly to capture context (not strictly necessary but helpful)
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
    classes = os.listdir(dataset_root)
    
    for class_name in classes:
        class_dir = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        all_imgs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle and pick N
        random.shuffle(all_imgs)
        selected = all_imgs[:n_images_per_class]
        image_paths.extend(selected)
        
    return image_paths
