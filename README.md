# Hybrid CNN–Mamba Framework for Automated Mycological Diagnosis

Production-ready deep learning framework for fungal infection classification from microscopic images using Hybrid State Space Models (Mamba) and Ensemble Learning.

---

## Overview

This repository implements a scalable AI system for automated fungal diagnosis from microscopic images.  
It combines convolutional feature extraction with linear-complexity State Space Models and transformer-based global reasoning.

Designed for:

- Clinical decision support systems
- High diagnostic reliability
- Real-time inference
- Edge-compatible deployment (hybrid model)

---

## Key Highlights

- **88.05% Accuracy** — Hybrid CNN–Mamba  
- **97.73% Accuracy** — Ensemble Model  
- **18.8 ms Inference Time** — Hybrid  
- **14.8 ms Inference Time** — Ensemble  
- Grad-CAM–based interpretability  
- GAN-based class balancing  

---

## Architecture

### Hybrid CNN–Mamba Model

EfficientNet-inspired convolutional encoder followed by Mamba State Space blocks for global context modeling.

**Pipeline**
Input (224×224)
→ CNN Encoder (MBConv + SE)
→ 1×1 Projection (1536 → 512)
→ Flatten to Sequence (49 × 512)
→ 2 × Mamba SSM Blocks
→ Global Average Pooling
→ Dropout (0.4)
→ Fully Connected Layer (5 classes)


**Why Mamba?**

- Captures long-range dependencies  
- Linear time complexity O(N)  
- Lower computational cost than Transformers  
- Suitable for medical imaging constraints  

---

### Ensemble Framework

Parallel multi-architecture fusion:

- Hybrid CNN–Mamba  
- EfficientNet-B3  
- ViT-B16  

Final prediction uses soft-voting over class probabilities.

This removes architectural bias and significantly improves reliability.

---

## Dataset

**DeFungi Dataset**

- 9,114 microscopic images  
- 5 clinically relevant fungal classes  

### Preprocessing

- Cropped to 500×500  
- Resized to 224×224  
- Zero-padding for aspect ratio preservation  
- On-the-fly augmentation:
  - Random rotations  
  - Horizontal & vertical flips  
  - Brightness / contrast adjustments  
- Class imbalance handled using DCGAN synthetic image generation  

---

## Training Configuration

| Parameter        | Value            |
|------------------|------------------|
| Optimizer        | AdamW            |
| Learning Rate    | 1e-4             |
| Weight Decay     | 1e-4             |
| Scheduler        | Cosine Annealing |
| Batch Size       | 32               |
| Epochs           | 40               |
| Early Stopping   | Patience = 8     |
| Loss Function    | Weighted Cross Entropy |

Dataset Split:
- 70% Training  
- 10% Validation  
- 20% Test  

---

## Performance Comparison

| Model                  | Params   | FLOPs        | Inference | Accuracy |
|------------------------|----------|-------------|-----------|----------|
| DenseNet121            | 5.1K     | 2.9 GFLOPs  | 47 ms     | 85.00%   |
| InceptionV3            | 10.1M    | 5.7 GFLOPs  | 12 ms     | 87.17%   |
| MobileNetV3-Small      | 0.35M    | 0.06 GFLOPs | 12 ms     | 79.28%   |
| MeFunX                 | 4M       | 0.77 GFLOPs | 30 ms     | 90.02%   |
| **CNN–Mamba**          | 14.87M   | 1.057 GFLOPs| 18.8 ms   | 88.05%   |
| **Ensemble (Hybrid)**  | 97.14M   | 18.323 GFLOPs| 14.8 ms  | 97.73%   |

---

## Interpretability

Grad-CAM visualizations confirm:

- Attention on hyphae structures  
- Focus on spores and colony morphology  
- Minimal background bias  

This supports clinical transparency and trust.

---

## Deployment Strategy

### Hybrid Model
- Lower FLOPs  
- Suitable for edge and resource-constrained systems  
- Balanced accuracy-efficiency tradeoff  

### Ensemble Model
- Highest diagnostic reliability  
- Recommended for hospital/server-grade infrastructure  

---

## Limitations

- Evaluated only on DeFungi dataset  
- Ensemble model has higher computational cost  
- Minor confusion between morphologically similar classes  

---

## Roadmap

- Cross-dataset validation  
- Domain adaptation  
- Self-supervised pretraining  
- Model compression for embedded deployment  

