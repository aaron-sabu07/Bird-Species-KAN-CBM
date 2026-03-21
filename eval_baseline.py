import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# 1. Dataset Loader (Reused for the Test Set)
class CUBFeatureDataset(Dataset):
    def __init__(self, split="test"):
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

# 2. Architecture (Must match exactly to load weights)
class BaselineCBM(nn.Module):
    def __init__(self, input_dim=512, num_concepts=312, num_classes=200):
        super().__init__()
        self.concept_predictor = nn.Linear(input_dim, num_concepts)
        self.label_predictor = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x):
        concept_logits = self.concept_predictor(x)
        concept_probs = torch.sigmoid(concept_logits)
        class_logits = self.label_predictor(concept_probs)
        return concept_logits, class_logits

# 3. Evaluation & NEC Logic
def evaluate_model():
    print("Loading Test Dataset and Model...")
    test_loader = DataLoader(CUBFeatureDataset("test"), batch_size=64, shuffle=False)
    
    model = BaselineCBM()
    model.load_state_dict(torch.load("vlg_cbm_sequential.pth"))
    model.eval()
    
    correct_preds = 0
    total_samples = len(test_loader.dataset)
    
    # Evaluate Accuracy
    with torch.no_grad():
        for x, _, y_true in test_loader:
            _, class_logits = model(x)
            correct_preds += (class_logits.argmax(dim=1) == y_true).sum().item()
            
    test_acc = correct_preds / total_samples
    print(f"\n--- TEST RESULTS ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # Calculate Number of Effective Concepts (NEC)
    print("\n--- CALCULATING NEC METRIC ---")
    # Extract the weights of the final linear layer (Shape: 200 classes x 312 concepts)
    weights = model.label_predictor.weight.detach().abs()
    
    # For each of the 200 bird classes, we find the maximum concept weight.
    # We consider a concept "Effective" if its weight is at least 10% of the maximum weight for that class.
    threshold = 0.10 
    
    nec_per_class = []
    for class_idx in range(200):
        class_weights = weights[class_idx]
        max_weight = class_weights.max()
        
        # Count how many concepts cross the significance threshold
        effective_concepts = (class_weights >= (threshold * max_weight)).sum().item()
        nec_per_class.append(effective_concepts)
        
    avg_nec = sum(nec_per_class) / len(nec_per_class)
    
    print(f"Average Number of Effective Concepts (NEC) per species: {avg_nec:.2f} / 312")
    print(f"Model Sparsity: {100 - ((avg_nec/312)*100):.2f}%")
    print("--------------------------------------------------")
    print("Note: In a standard linear model, NEC is often very high. Our goal with the KAN-CBM ")
    print("will be to drastically reduce this NEC while maintaining or improving accuracy.")

if __name__ == "__main__":
    evaluate_model()