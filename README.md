<div align="center">

# 🎭 Masked Autoencoder (MAE)
### Self-Supervised Image Representation Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Dataset](https://img.shields.io/badge/Dataset-TinyImageNet-orange)](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

<br/>

*A full PyTorch implementation of the **Masked Autoencoder (MAE)** paper — learning powerful visual representations by reconstructing images from sparse, visible patches using an asymmetric ViT encoder-decoder.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Streamlit App](#-streamlit-app)
- [Results](#-results)
- [Assignment Context](#-assignment-context)
- [License](#-license)

---

## 🧠 Overview

Masked Autoencoders (MAE) are a form of **self-supervised learning** for computer vision, inspired by BERT's masked language modeling. The key idea is simple yet powerful:

> **Mask 75% of an image's patches at random → train a model to reconstruct them.**

This forces the model to develop a deep understanding of image structure and semantics without any labeled data. The learned representations can then be fine-tuned for downstream tasks.

This implementation follows the original **asymmetric encoder-decoder design** from [He et al., 2021](https://arxiv.org/abs/2111.06377):
- The **large ViT-Base encoder** only processes the visible 25% of patches.
- The **lightweight ViT-Small decoder** receives encoded visible tokens + learnable mask tokens and reconstructs all patches.

---

## 🏗️ Architecture

```
Input Image (224×224)
        │
        ▼
  ┌─────────────┐
  │  Patchify   │  → 196 patches of 16×16
  └─────────────┘
        │
        ▼  Random Masking (75%)
  ┌─────────────────────────────┐
  │  25% Visible Patches (49)   │  + Positional Embeddings
  └─────────────────────────────┘
        │
        ▼
  ┌────────────────────────┐
  │   ENCODER  (ViT-Base)  │  d=768, 12 layers, 12 heads, ~86M params
  └────────────────────────┘
        │  Latent tokens (visible only)
        ▼
  ┌───────────────────────────────────────────┐
  │  Projection (768 → 384) + Mask Tokens     │
  └───────────────────────────────────────────┘
        │  Full sequence (196 tokens)
        ▼
  ┌─────────────────────────┐
  │  DECODER  (ViT-Small)   │  d=384, 12 layers, 6 heads, ~22M params
  └─────────────────────────┘
        │
        ▼
  ┌─────────────┐
  │  Prediction │  Linear → 16×16×3 per patch
  └─────────────┘
        │  MSE Loss on MASKED patches only
        ▼
  Reconstructed Image (224×224)
```

### Encoder — ViT-Base/16

| Parameter         | Value   |
|-------------------|---------|
| Image Size        | 224×224 |
| Patch Size        | 16×16   |
| Hidden Dim        | 768     |
| Transformer Layers| 12      |
| Attention Heads   | 12      |
| FFN Dim           | 3072    |
| Parameters        | ~86M    |
| Input             | 25% visible patches only |

### Decoder — ViT-Small/16

| Parameter         | Value   |
|-------------------|---------|
| Hidden Dim        | 384     |
| Transformer Layers| 12      |
| Attention Heads   | 6       |
| FFN Dim           | 1536    |
| Parameters        | ~22M    |
| Input             | Encoded visible tokens + learnable mask tokens |

---

## 📁 Project Structure

```
📦 Self-Supervised Image Representation Learning - MAE
 ┣ 📂 Model
 ┃ ┣ 📄 model_lib.py          # MAE model definition (PatchEmbed, MaskedAutoencoder)
 ┃ ┗ 📄 mae_weights.pth       # Pre-trained model weights (~411 MB)
 ┣ 📂 Notebook
 ┃ ┗ 📓 MAE_Training.ipynb    # Full Kaggle training notebook
 ┣ 📄 app.py                  # Streamlit interactive app
 ┣ 📄 requirements.txt        # Python dependencies
 ┣ 📄 LICENSE                 # MIT License
 ┗ 📄 README.md               # You are here
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pillow](https://img.shields.io/badge/-Pillow-3776AB?style=flat-square&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![CUDA](https://img.shields.io/badge/-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU with CUDA support (recommended) or CPU

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mae-image-representation.git
   cd mae-image-representation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the model weights** are present at `Model/mae_weights.pth`.

---

## 🏋️ Training

Training was performed on **Kaggle** using **Dual T4 GPUs** on the **TinyImageNet** dataset.

### Key Training Configuration

| Setting               | Value              |
|-----------------------|--------------------|
| Dataset               | TinyImageNet (200 classes, 100K images) |
| Accelerator           | Kaggle T4 × 2 (Dual GPU) |
| Masking Ratio         | 75%                |
| Batch Size            | 32–64              |
| Optimizer             | AdamW              |
| LR Scheduler          | Cosine Annealing   |
| Precision             | Mixed (AMP)        |
| Loss Function         | MSE (masked patches only) |

### Run the Notebook

Open `Notebook/MAE_Training.ipynb` on Kaggle with:
- **Accelerator**: GPU T4 × 2
- **Dataset**: [TinyImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) attached

The notebook covers:
- ✅ Data loading & augmentation pipeline
- ✅ MAE model initialization
- ✅ Training loop with Mixed Precision (AMP)
- ✅ Loss curve visualization
- ✅ Qualitative reconstruction samples (≥5)
- ✅ PSNR & SSIM quantitative evaluation

---

## 🖥️ Streamlit App

An interactive web application lets you upload any image, adjust the masking ratio, and observe the MAE reconstruction in real time.

### Launch the App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### App Features

| Feature                    | Description                                      |
|----------------------------|--------------------------------------------------|
| 📤 Image Upload            | Accepts JPG, JPEG, PNG images                    |
| 🎚️ Masking Ratio Slider   | Dynamically adjust masking from 10% to 90%       |
| 🖼️ Side-by-Side Display   | Shows **Masked Input**, **Reconstruction**, and **Original GT** |
| ⚡ Real-time Inference     | Instant reconstruction on CPU/GPU                |

---

## 📊 Results

### Qualitative Reconstructions

The model demonstrates strong reconstruction ability even at 75% masking, capturing textures, object boundaries, and semantic regions effectively.

| View             | Description                            |
|------------------|----------------------------------------|
| Masked Input     | 75% of patches zeroed out              |
| MAE Reconstruction | Model-inferred pixel content         |
| Ground Truth     | Original unmodified image              |

### Quantitative Metrics

| Metric | Description                                          |
|--------|------------------------------------------------------|
| PSNR   | Peak Signal-to-Noise Ratio (higher is better, in dB) |
| SSIM   | Structural Similarity Index (1.0 = perfect)          |

*Full metrics are reported in the training notebook.*

---

## 📚 Assignment Context

> **Course**: Generative AI — AI4009  
> **Institution**: National University of Computer and Emerging Sciences (FAST-NUCES)  
> **Semester**: Spring 2026  
> **Assignment**: No. 2 — Self-Supervised Image Representation Learning using Masked Autoencoders

This project implements all required deliverables:
- [x] Asymmetric ViT encoder-decoder architecture from scratch in pure PyTorch
- [x] Patch-level masking with 75% ratio
- [x] MSE loss computed only on masked patches
- [x] AdamW + Cosine LR Scheduler
- [x] Mixed Precision (AMP) training on Dual T4 GPUs
- [x] ≥5 qualitative reconstruction examples
- [x] PSNR & SSIM metrics
- [x] Streamlit deployment app

---

## 📖 Reference

> Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.  
> **Masked Autoencoders Are Scalable Vision Learners** (2021).  
> [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  Made with ❤️ using PyTorch &nbsp;|&nbsp; FAST-NUCES Spring 2026
</div>
