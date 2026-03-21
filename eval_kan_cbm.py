import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from efficient_kan import KAN

# 1. Dataset Loader
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

# 2. Architecture Definition
class KAN_CBM_VIB(nn.Module):
    def __init__(self, input_dim=512, num_concepts=312, num_classes=200):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, num_concepts)
        self.fc_logvar = nn.Linear(input_dim, num_concepts)
        self.label_predictor = KAN([num_concepts, num_classes])
        
    def forward(self, x):
        mu = self.fc_mu(x)
        concept_probs = torch.sigmoid(mu) # Deterministic for evaluation
        class_logits = self.label_predictor(concept_probs)
        return concept_probs, class_logits

# 3. Evaluation & KAN-NEC Logic
def evaluate_kan():
    print("Loading Test Dataset and KAN-CBM Model...")
    test_loader = DataLoader(CUBFeatureDataset("test"), batch_size=64, shuffle=False)
    
    model = KAN_CBM_VIB()
    model.load_state_dict(torch.load("kan_cbm_sparse.pth"), strict=False)
    model.eval()
    
    correct_preds = 0
    total_samples = len(test_loader.dataset)
    
    # 1. Evaluate Test Accuracy
    with torch.no_grad():
        for x, _, y_true in test_loader:
            _, class_logits = model(x)
            correct_preds += (class_logits.argmax(dim=1) == y_true).sum().item()
            
    test_acc = correct_preds / total_samples
    print(f"\n--- TEST RESULTS ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # 2. Calculate NEC for Kolmogorov-Arnold Splines
    print("\n--- CALCULATING KAN-NEC METRIC ---")
    
    # Access the first (and only) KAN layer
    kan_layer = model.label_predictor.layers[0]
    
    # A KAN edge weight is a combination of its base linear weight and the average magnitude of its spline control points
    base_weights = kan_layer.base_weight.detach().abs()
    spline_weights = kan_layer.spline_weight.detach().abs().mean(dim=2)
    
    # Total effective magnitude matrix (200 classes x 312 concepts)
    total_effective_weights = base_weights + spline_weights
    
    threshold = 0.10 
    nec_per_class = []
    
    for class_idx in range(200):
        class_weights = total_effective_weights[class_idx]
        max_weight = class_weights.max()
        effective_concepts = (class_weights >= (threshold * max_weight)).sum().item()
        nec_per_class.append(effective_concepts)
        
    avg_nec = sum(nec_per_class) / len(nec_per_class)
    
    print(f"Average Number of Effective Concepts (NEC) per species: {avg_nec:.2f} / 312")
    print(f"Model Sparsity: {100 - ((avg_nec/312)*100):.2f}%")
    print("--------------------------------------------------")

if __name__ == "__main__":
    evaluate_kan()