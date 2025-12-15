import torch
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import utils


#hyperparameters
DATASET_PATH = "data/imagenette2/train" # Or "data/CUB_200_2011/train"
NUM_CLUSTERS = 2048             # Paper uses 2048. Use 512 for quick testing.
IMGS_PER_CLASS = 50             # Paper suggests 50 is enough
MIN_AREA = 1000                 # Filter out tiny segments (pixels^2)
BATCH_SIZE = 32                 # For CLIP encoding
OUTPUT_FILE = "artifacts/concepts/concept_bank.pt"



def main():
    os.makedirs("artofacts/concepts", exist_ok=True)
    device = utils.get_device()

    # load models
    clip_model = utils.load_models()
    clip_preprocess = utils.load_models()
    mask_generator = utils.load_models()


    # get data list
    print(f"Selecting {IMGS_PER_CLASS} iages per class from {DATASET_PATH}...")
    image_paths = utils.get_image_paths(DATASET_PATH, n_images_per_class= IMGS_PER_CLASS)
    print(f"Total images to process: {len(image_paths)}")

    
    # STEP 1: Segmentation and Embedding
    all_crop_embeddings = []

    with torch.no_grad():
        for img_path in tqdm(image_paths):
            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image) #SAM requires an array


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
                    
                
                # Batch encode crops with CLIP
                crop_tensor = torch.stack(valid_crops).to(device)

                # encode image -> [batch size, 512]
                features = clip_model.encode_image(crop_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


    # concatenate all embeddings into one large matrrix
    if len(all_crop_embeddings) == 0:
        print("Error: No segments found! Check your data path.")
        return
    

    full_embedding_matrix = torch.cat(all_crop_embeddings, dim=0).numpy()

    print(f"Extracted {full_embedding_matrix.shape[0]} segment embeddings. We now cluster into {NUM_CLUSTERS} concepts...")


    # STEP 2: Clustering using K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=10, random_state=42)
    kmeans.fit(full_embedding_matrix)

    # remember, the centroids are our "Concept Bank"
    # shape: [NUM_CLUSTERS, 512]

    centroids = kmeans.cluster_centers_

    #convert it back to torch tensor
    concept_bank = torch.from_numpy(centroids).float()

    torch.save(concept_bank, OUTPUT_FILE)
    print(f"Concept bank saved to {OUTPUT_FILE}")
    print(f"Shape: {concept_bank.shape}")



if __name__ == "__main__":
    main()

    
