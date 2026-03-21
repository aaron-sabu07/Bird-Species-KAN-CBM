import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# 1. Dataset Loader for our .npy features
class CUBFeatureDataset(Dataset):
    def __init__(self, split="train"):
        base_dir = "data/features"
        self.features = np.load(os.path.join(base_dir, f"features_{split}.npy"))
        self.labels = np.load(os.path.join(base_dir, f"labels_{split}.npy"))
        self.concepts = np.load(os.path.join(base_dir, f"concepts_{split}.npy"))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        c = torch.FloatTensor(self.concepts[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, c, y

# 2. The VLG-CBM Architecture
class BaselineCBM(nn.Module):
    def __init__(self, input_dim=512, num_concepts=312, num_classes=200):
        super().__init__()
        # g(x) -> c: Maps CLIP features to Bird Attributes
        self.concept_predictor = nn.Linear(input_dim, num_concepts)
        # f(c) -> y: Maps Bird Attributes to Species
        self.label_predictor = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x):
        # Predict concept logits
        concept_logits = self.concept_predictor(x)
        # Squeeze through sigmoid to get probabilities [0, 1]
        concept_probs = torch.sigmoid(concept_logits)
        # Predict species strictly from concepts
        class_logits = self.label_predictor(concept_probs)
        return concept_logits, class_logits

# 3. Training Setup
def train_model():
    # Setup DataLoaders
    train_loader = DataLoader(CUBFeatureDataset("train"), batch_size=64, shuffle=True)
    test_loader = DataLoader(CUBFeatureDataset("test"), batch_size=64, shuffle=False)
    
    model = BaselineCBM()
    
    # Joint Loss Definition
    concept_criterion = nn.BCEWithLogitsLoss()
    label_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    lambda_weight = 5.0 # Weight for concept loss
    epochs = 60
    
    print("Starting Training: VLG-CBM Baseline...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct_preds = 0, 0
        
        for x, c_true, y_true in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            concept_logits, class_logits = model(x)
            
            # Calculate Losses
            loss_c = concept_criterion(concept_logits, c_true)
            loss_y = label_criterion(class_logits, y_true)
            
            # Joint Loss Formula
            loss = loss_y + (lambda_weight * loss_c)
            
            # Backward and Optimize
            loss.backward()
            optimizer.step()
            
            total_loss = total_loss + loss.item() # Note: Out-of-place addition
            correct_preds = correct_preds + (class_logits.argmax(dim=1) == y_true).sum().item()
            
        acc = correct_preds / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc*100:.2f}%")

if __name__ == "__main__":
    train_model()