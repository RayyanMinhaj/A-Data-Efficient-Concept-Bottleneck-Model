import torch
import torch.nn as nn
import clip
from PIL import Image
import os
import utils


# We will grab a random validation image to test
TEST_IMAGE_PATH = "data/imagenette2/val/n03425413/ILSVRC2012_val_00000732.JPEG" 

CONCEPT_BANK_PATH = "artifacts/concepts/concept_bank.pt"
MODEL_PATH = "artifacts/models/dcbm_model.pth"
DATASET_ROOT = "data/imagenette2/train"

# A mini vocabulary for the demo. In a real app, use a dictionary of 10k+ words.
VOCABULARY = [
    "fish", "scale", "fin", "water", "river", "net", "mesh",
    "dog", "fur", "ear", "paw", "tail", "spaniel", "eye",
    "radio", "speaker", "button", "plastic", "metal", "music",
    "saw", "blade", "handle", "orange", "yellow", "chain", "tool",
    "church", "building", "window", "brick", "cross", "sky",
    "horn", "brass", "gold", "instrument", "mouth",
    "truck", "wheel", "tire", "garbage", "green", "road",
    "gas", "pump", "red", "hose", "nozzle", "station",
    "golf", "ball", "white", "grass", "dimple", "sport",
    "parachute", "cord", "fabric", "air", "jump", "cloud",
    "wood", "tree", "leaf", "blue", "black", "glass"
]

class DCBM(nn.Module):
    def __init__(self, concept_bank, num_classes, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        self.concept_bank = concept_bank.to(device)
        self.concept_bank = self.concept_bank / self.concept_bank.norm(dim=-1, keepdim=True)
        num_concepts = self.concept_bank.shape[0]
        self.classifier = nn.Linear(num_concepts, num_classes, bias=False)



    def forward(self, x):
        with torch.no_grad():
            img_features = self.clip_model.encode_image(x).float()
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            concept_activations = torch.matmul(img_features, self.concept_bank.T)
        logits = self.classifier(concept_activations)
        return logits, concept_activations
    


    
def name_concepts(concept_bank, vocabulary, device):
    """
    Step 4: Uses CLIP Text Encoder to give names to our visual concepts.
    """
    print(f"Naming {concept_bank.shape[0]} concepts using {len(vocabulary)} words...")
    
    # 1. Load CLIP
    model, _ = clip.load("ViT-B/16", device=device)
    
    # 2. Encode all words in vocabulary
    text_inputs = clip.tokenize(vocabulary).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 3. Match Visual Concepts to Text
    # Matrix Mul: [num_concepts, 512] @ [512, num_words] -> [num_concepts, num_words]
    similarity = torch.matmul(concept_bank, text_features.T)
    
    # 4. Find best word for each concept
    best_word_indices = similarity.argmax(dim=1)
    
    concept_names = [vocabulary[idx] for idx in best_word_indices]
    return concept_names



def main():
    device = utils.get_device()
    
    # 1. Load Metadata
    import torchvision.datasets as dset
    # We load the dataset just to get the class names correctly
    temp_data = dset.ImageFolder(root=DATASET_ROOT)
    class_names = temp_data.classes
    
    # 2. Load Model & Concepts
    print("Loading Model...")
    bank_tensor = torch.load(CONCEPT_BANK_PATH, map_location=device)
    model = DCBM(bank_tensor, len(class_names), device).to(device)
    
    # Load trained weights
    # strict=False allows ignoring CLIP weights if they weren't saved perfectly, 
    # but usually strictly loading is better.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    
    # 3. Step 4: Name the Concepts
    concept_names = name_concepts(bank_tensor, VOCABULARY, device)
    
    # 4. Get an Image
    _, preprocess = clip.load("ViT-B/16", device=device)
    
    img_path = TEST_IMAGE_PATH
    if not os.path.exists(img_path):
        # Pick random if hardcoded one is missing
        import random
        all_imgs = utils.get_image_paths("data/imagenette2/val", n_images_per_class=1)
        img_path = random.choice(all_imgs)
        
    print(f"\nExplaining Image: {img_path}")
    image = Image.open(img_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        logits, activations = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get Top Prediction
        top_prob, top_class_idx = probs.max(dim=1)
        predicted_class = class_names[top_class_idx]
        
        print(f"Prediction: {predicted_class} ({top_prob.item()*100:.2f}%)")
        print("-" * 30)
        
        # 6. EXPLANATION LOGIC
        # We want to know: Which concepts contributed most to THIS class prediction?
        # Contribution = Activation_of_Concept * Weight_in_Linear_Layer
        
        # Get weights for the predicted class [num_concepts]
        class_weights = model.classifier.weight[top_class_idx].squeeze() 
        
        # Get activations for this image [num_concepts]
        current_activations = activations.squeeze()
        
        # Calculate contribution
        contributions = class_weights * current_activations
        
        # Get Top 5 Concepts
        top_indices = torch.argsort(contributions, descending=True)[:5]
        
        print(f"Why? Because I saw:")
        for idx in top_indices:
            concept_id = idx.item()
            name = concept_names[concept_id]
            score = current_activations[concept_id].item()
            contrib = contributions[concept_id].item()
            
            print(f" - Concept {concept_id:04d} ('{name}'): Activation {score:.2f} (Impact: {contrib:.2f})")










if __name__ == "__main__":
    main()
