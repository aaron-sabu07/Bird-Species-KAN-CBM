# 🦜 Decoding Nature: Phase I
### Interpretable VLG-CBMs for Avian Species Classification

---

## 📌 Project Overview

This repository contains the implementation of the **Orange Problem (Phase I)** for the XAI course. The goal is to build a **"Faithful Glass-Box"** architecture for fine-grained bird species classification using the **CUB-200-2011** dataset. By utilizing a Concept Bottleneck Model (CBM) framework, the system translates high-dimensional visual data into **312 human-readable biological attributes** before predicting the species label.

---

## 🏗️ Architecture: The Glass-Box Pipeline

Instead of a standard black-box $x \to y$ mapping, this model utilizes a structured $x \to c \to y$ reasoning chain:

| Step | Component | Description |
|------|-----------|-------------|
| 1 | **Feature Extractor** | Frozen CLIP-ViT-B/16 backbone converts images into $512$-dimensional embeddings |
| 2 | **Concept Bottleneck** | A linear layer predicts 312 ornithological concepts (e.g., *"hooked beak"*, *"blue wing"*) |
| 3 | **Reasoning Head** | A species classifier (200 classes) that operates strictly on predicted concept probabilities |

---

## 🔬 Methodology: Sequential Training

To ensure the purity of the bottleneck and prevent **"concept leakage,"** we implemented a **Sequential Training** strategy:

- **Phase 1:** Train the concept predictor using Multi-label Binary Cross-Entropy (BCE).
- **Phase 2:** Freeze the concept predictor and train the label predictor using Cross-Entropy (CE).

---

## 📊 Performance Results

The model was evaluated on both classification accuracy and the **Number of Effective Concepts (NEC)** — a mathematical measure of sparsity and information leakage.

| Metric | Result |
|--------|--------|
| 🏋️ Training Accuracy | **88.66%** |
| 🧪 Test Accuracy | **79.17%** |
| 🔢 Avg. NEC | **230.24 / 312** |
| 🌿 Model Sparsity | **26.21%** |

> **Key Finding:** While the model achieves high accuracy (~79%), the high NEC score indicates that the linear layer is relying on too many concepts for a single decision, suggesting a need for non-linear, sparse architectures *(Phase II: Jackfruit Problem)*.

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
python -m venv xai_env
source xai_env/bin/activate  # .\xai_env\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Data Preparation

Extract CLIP features using the provided Kaggle script and place `.npy` files in `data/features/`.

### 3. Training & Evaluation

```bash
python train_baseline_seq.py
python eval_baseline.py
```

---

## 📄 Acknowledgments

This research is based on the foundational work of:

- **Koh et al. (2020)** on Concept Bottleneck Models
- **Srivastava et al. (2025)** on Vision-Language Guided CBMs