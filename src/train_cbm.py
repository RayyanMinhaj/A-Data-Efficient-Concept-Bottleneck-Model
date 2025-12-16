import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
import os
import utils


DATASET_ROOT = "data/imagenette2"  
CONCEPT_BANK_PATH = "artifacts/concepts/concept_bank.pt"
MODEL_SAVE_PATH = "artifacts/models/dcbm_model.pth"

BATCH_SIZE = 128
EPOCHS = 20           # It converges fast because we only train one layer
LEARNING_RATE = 0.01  # Higher LR is okay for linear layers
LAMBDA_SPARSITY = 0.0001 # The 'L1' penalty strength (Paper suggests 1e-4)



class DCBM(nn.Module):
    def __init__(self, concept_bank, num_classes, device):
        super().__init__()

        self.device = device

        # Setup CLIP (frozen) because we dont need the preprocessing part (Dataloader will handel it)
        self.clip_model, _ = clip.load("ViT-B/16", device=device)

        for param in self.clip_model.parameters():
            param.requires_grad = False

        
        # Setup concept bank (frozen)
        self.concept_bank = concept_bank.to(device)
        self.concept_bank.requires_grad = False

        # Normalize concept bank to unit length fro cosine similarity
        self.concept_bank = self.concept_bank / self.concept_bank.norm(dim=-1, keepdim=True)

        num_concepts = self.concept_bank.shape[0]

        # The Bottleneck Layer (Trainable)
        # Input: Concept Scores -> Output: Class Logits
        # bias=False helps interpretability (0 concept activation = 0 contribution)
        self.classifier = nn.Linear(num_concepts, num_classes, bias=False)

    

    def forward(self, x):
        # remember x is [batch size, 3, 244, 244]

        with torch.no_grad():
            img_features = self.clip_model.encode_image(x)  
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # This is the blue matrix from Paper (Image @ Concept Bank^T)
            concept_activations = torch.matmul(img_features, self.concept_bank.T)  # [batch size, num concepts]
        
        logits = self.classifier(concept_activations)

        return logits, concept_activations



def main():
    device = utils.get_device()
    os.makedirs("artifacts/models", exist_ok=True)

    _, preprocess_fn = clip.load("ViT-B/16", device=device) # we only want thse CLIP transform

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_ROOT, 'train'),
        trasnform = preprocess_fn
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_ROOT, 'val'),
        transform=preprocess_fn
    )


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    # Load concept bank
    if not os.path.exists(CONCEPT_BANK_PATH):
        print(f"Error: Concept bank not found at {CONCEPT_BANK_PATH}")
        return
        
    print("Loading Concept Bank...")

    bank_tensor = torch.load(CONCEPT_BANK_PATH)

    print(f"Concepts Loaded: {bank_tensor.shape[0]}")


    # Initialize DCBM model
    model = DCBM(bank_tensor, len(class_names), device).to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    print(f"Starting Training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits, _ = model(images)
            
            # Loss Calculation
            ce_loss = criterion(logits, labels)
            
            # Add L1 Sparsity Penalty (Sum of absolute values of weights)
            # This encourages the model to use fewer concepts
            l1_norm = torch.norm(model.classifier.weight, p=1)
            loss = ce_loss + (LAMBDA_SPARSITY * l1_norm)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")


    
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    #print("Ready for Step 4 (Inference & Naming)!")



if __name__ == "__main__":
    main()
