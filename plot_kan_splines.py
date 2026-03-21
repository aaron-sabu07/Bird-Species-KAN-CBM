import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from efficient_kan import KAN

# 1. Architecture Definition
class KAN_CBM_VIB(nn.Module):
    def __init__(self, input_dim=512, num_concepts=312, num_classes=200):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, num_concepts)
        self.fc_logvar = nn.Linear(input_dim, num_concepts)
        self.label_predictor = KAN([num_concepts, num_classes])
        
    def forward(self, x):
        pass # Not needed for isolated spline extraction

def load_concept_names():
    """Loads human-readable concept names from the CUB dataset."""
    try:
        base_path = "data/attributes.txt"
        if not os.path.exists(base_path):
            # Fallback path based on earlier Kaggle extraction
            base_path = "data/attributes/attributes.txt" 
            
        df = pd.read_csv(base_path, sep=r'\s+', names=['id', 'attribute_name'], engine='python')
        # Clean up the names (e.g., 'has_bill_shape::dagger' -> 'Bill Shape: Dagger')
        names = df['attribute_name'].apply(lambda x: x.split('::')[-1].replace('_', ' ').title()).tolist()
        return names
    except Exception as e:
        print("Could not load concept names, using indices instead.")
        return [f"Concept {i}" for i in range(312)]

def plot_splines_for_species(model, target_class=0, num_top_concepts=4):
    print(f"Extracting KAN Splines for Bird Species ID: {target_class}")
    concept_names = load_concept_names()
    
    # 1. Identify the most critical concepts for this specific bird using our NEC logic
    kan_layer = model.label_predictor.layers[0]
    base_weights = kan_layer.base_weight.detach().abs()
    spline_weights = kan_layer.spline_weight.detach().abs().mean(dim=2)
    total_effective_weights = base_weights + spline_weights
    
    # Get the indices of the highest weighted concepts for this class
    class_weights = total_effective_weights[target_class]
    top_concept_indices = torch.topk(class_weights, num_top_concepts).indices.tolist()
    
    # 2. Setup the Plot
    fig, axes = plt.subplots(1, num_top_concepts, figsize=(16, 4))
    fig.suptitle(f"KAN Symbolic Discovery: Top Biological Traits for Species {target_class}", fontsize=16, fontweight='bold', y=1.05)
    
    # Generate a sweep of probabilities from 0.0 (Concept Absent) to 1.0 (Concept Present)
    x_sweep = torch.linspace(0, 1, 100)
    
    for i, concept_idx in enumerate(top_concept_indices):
        # Create an empty input matrix (100 steps, 312 concepts)
        isolated_input = torch.zeros((100, 312))
        
        # Inject our sweeping probability ONLY into the target concept
        isolated_input[:, concept_idx] = x_sweep
        
        # Pass through the KAN head
        with torch.no_grad():
            logits = model.label_predictor(isolated_input)
            
        # Extract the resulting prediction curve for our target bird species
        y_curve = logits[:, target_class].numpy()
        
        # Normalize the curve to start at 0 for clean visualization (relative contribution)
        y_curve = y_curve - y_curve[0] 
        
        # Plotting
        ax = axes[i]
        ax.plot(x_sweep.numpy(), y_curve, color='#2ca02c', linewidth=3)
        ax.fill_between(x_sweep.numpy(), y_curve, 0, color='#2ca02c', alpha=0.2)
        
        trait_name = concept_names[concept_idx] if concept_idx < len(concept_names) else f"Concept {concept_idx}"
        ax.set_title(f"Trait: {trait_name}", fontsize=12)
        ax.set_xlabel("Concept Probability", fontsize=10)
        ax.set_ylabel("Contribution to Species Logit", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"kan_spline_species_{target_class}.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to kan_spline_species_{target_class}.png")
    plt.show()

if __name__ == "__main__":
    # Load the sparse weights
    model = KAN_CBM_VIB()
    model.load_state_dict(torch.load("kan_cbm_sparse.pth"), strict=False)
    model.eval()
    
    # Generate plots for two different bird species to show contrasting logic
    plot_splines_for_species(model, target_class=10)  # E.g., Red-winged Blackbird
    plot_splines_for_species(model, target_class=150) # E.g., Sage Thrasher