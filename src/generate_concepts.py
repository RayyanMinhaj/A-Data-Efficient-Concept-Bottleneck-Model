import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import utils


DATASET_PATH = "/mnt/sdz/rami01/Desktop/DCBM_PROJECT/A-Data-Efficient-Concept-Bottleneck-Model/data/imagenette2/train" 
NUM_CLUSTERS = 512              # Lowered to 512 for faster debugging
IMGS_PER_CLASS = 50             
MIN_AREA = 1000                 
OUTPUT_FILE = "artifacts/concepts/concept_bank.pt"




def main():

    os.makedirs("artifacts/concepts", exist_ok=True)
    device = utils.get_device()
    
   # load models
    clip_model, clip_preprocess, mask_generator = utils.load_models()

    # --- DIAGNOSTIC CHECK ---
    # This prevents the script from crashing 
    print(f"DIAGNOSTIC: mask_generator is type: {type(mask_generator)}")
    if not hasattr(mask_generator, 'generate'):
        print("CRITICAL ERROR: The loaded mask_generator does not have a .generate() method.")
        print("This usually means utils.py is returning the wrong object.")
        return
    # ------------------------
    
    # get data list
    image_paths = utils.get_image_paths(DATASET_PATH, n_images_per_class=IMGS_PER_CLASS)
    if not image_paths:
        print(f"No images found in {DATASET_PATH}. Please check the path.")
        return
    print(f"Total images to process: {len(image_paths)}")

    
    
    # STEP 1: Segmentation and Embedding
    all_crop_embeddings = []

    print("Starting Segmentation and Embedding...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image) # SAM requires array
                
                # generate masks for SAM
                masks = mask_generator.generate(image_np)
                
     
                valid_crops = []
                for mask in masks:
                    if mask['area'] < MIN_AREA:
                        continue
                    
                    crop = utils.crop_image_from_mask(image, mask)
                    preprocessed_crop = clip_preprocess(crop)
                    valid_crops.append(preprocessed_crop)
                
                if not valid_crops:
                    continue

                # batch Encode with CLIP
                crop_tensor = torch.stack(valid_crops).to(device)

                # encode image -> [batch size, 512]
                features = clip_model.encode_image(crop_tensor)
                features /= features.norm(dim=-1, keepdim=True) 
                
                all_crop_embeddings.append(features.cpu())
                
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")
                continue

    
    # concatenate all embeddings into one large matrrix
    if len(all_crop_embeddings) == 0:
        print("No segments collected. Aborting.")
        return

    full_embedding_matrix = torch.cat(all_crop_embeddings, dim=0).numpy()
    print(f"Extracted {full_embedding_matrix.shape[0]} total segments.")




    # STEP 2: Clustering using K-Means
    print(f"Clustering into {NUM_CLUSTERS} concepts using K-Means...")
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(full_embedding_matrix)
    
    centroids = kmeans.cluster_centers_
    concept_bank = torch.from_numpy(centroids).float()
    
    torch.save(concept_bank, OUTPUT_FILE)
    print(f"Success! Concept bank saved to {OUTPUT_FILE}")




if __name__ == "__main__":
    main()
