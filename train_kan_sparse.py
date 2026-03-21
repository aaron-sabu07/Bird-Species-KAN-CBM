import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from efficient_kan import KAN

torch.manual_seed(42)

# 1. Dataset Loader
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

# 2. Architecture
class KAN_CBM_VIB(nn.Module):
    def __init__(self, input_dim=512, num_concepts=312, num_classes=200):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, num_concepts)
        self.fc_logvar = nn.Linear(input_dim, num_concepts)
        self.label_predictor = KAN([num_concepts, num_classes])
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps.clone() * std
        
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        concept_probs = torch.sigmoid(z)
        class_logits = self.label_predictor(concept_probs)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return concept_probs, class_logits, kl_loss

# 3. The Sparse Training Protocol
def train_kan_sparse():
    train_loader = DataLoader(CUBFeatureDataset("train"), batch_size=64, shuffle=True)
    model = KAN_CBM_VIB()
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = CosineAnnealingLR(optimizer, T_max=40)
    criterion_y = nn.CrossEntropyLoss()
    
    # Adjusted Hyperparameters
    lambda_c = 5.0    
    beta_vib = 0.005  # Increased by 5x to squeeze the bottleneck harder
    lambda_l1 = 1e-4  # The new L1 penalty multiplier
    epochs = 40
    
    print("--- Initiating Sparse Jackfruit Protocol: KAN-CBM with L1 & Aggressive VIB ---")
    for epoch in range(epochs):
        model.train()
        total_loss, correct_preds = 0, 0
        
        for x, c_true, y_true in train_loader:
            optimizer.zero_grad()
            
            concept_probs, class_logits, kl_loss = model(x)
            loss_c = nn.BCELoss()(concept_probs, c_true) 
            loss_y = criterion_y(class_logits, y_true)
            
            # --- THE L1 SPLINE PENALTY ---
            # We extract the specific KAN layer to penalize its exact weights
            kan_layer = model.label_predictor.layers[0]
            l1_loss = kan_layer.base_weight.abs().sum() + kan_layer.spline_weight.abs().sum()
            
            # The Unified Objective Function (With 4 constraints)
            loss = loss_y + (lambda_c * loss_c) + (beta_vib * kl_loss) + (lambda_l1 * l1_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss = total_loss + loss.item()
            correct_preds = correct_preds + (class_logits.argmax(dim=1) == y_true).sum().item()
            
        scheduler.step()
        acc = correct_preds / len(train_loader.dataset)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc*100:.2f}% | KL: {kl_loss.item():.4f}")
            
    torch.save(model.state_dict(), "kan_cbm_sparse.pth")
    print("\nModel saved as kan_cbm_sparse.pth")

if __name__ == "__main__":
    train_kan_sparse()