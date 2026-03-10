# Masked Autoencoder (MAE) for Image Reconstruction

A PyTorch implementation of a **Masked Autoencoder (MAE)** based on Vision Transformers (ViT) for self-supervised image representation learning.  
This project trains a model to reconstruct masked patches of images, enabling efficient feature learning without labeled data.

---

## Overview

Masked Autoencoders (MAE) are a self-supervised learning technique where a large portion of image patches are masked and the model learns to reconstruct the missing information.

In this project:

- Images are divided into **patches**
- **75% of patches are masked**
- A **Vision Transformer encoder** processes the visible patches
- A **lightweight decoder** reconstructs the full image
- Reconstruction quality is evaluated using **PSNR** and **SSIM**

---

## Key Features

- Vision Transformer based **MAE Encoder**
- Patch embedding with positional encoding
- Custom **Multi-Head Self Attention**
- Transformer blocks for deep feature learning
- Image reconstruction via MAE Decoder
- Mixed precision training with **PyTorch AMP**
- Cosine learning rate scheduler with warmup
- Visualization of reconstructed images
- Quantitative evaluation using **PSNR and SSIM**
- Interactive **Gradio web interface** for inference

---

## Project Architecture

```
Input Image
      │
Patch Embedding (16x16)
      │
Masking (75% patches removed)
      │
MAE Encoder (Vision Transformer)
      │
Latent Representation
      │
MAE Decoder
      │
Reconstructed Image
```

---

## Model Components

### Patch Embedding
Converts the image into fixed-size patches and embeds them into vector representations.

### Positional Encoding
Uses **2D sine-cosine positional embeddings** to maintain spatial information.

### Transformer Encoder
Processes visible patches using:

- Multi-head self-attention
- Feed-forward networks
- Layer normalization

### MAE Decoder
Reconstructs masked patches from encoded features.

---

## Configuration

Main training parameters:

| Parameter | Value |
|----------|------|
| Image Size | 224 |
| Patch Size | 16 |
| Number of Patches | 196 |
| Mask Ratio | 0.75 |
| Encoder Dimension | 768 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |

---

## Training Pipeline

1. Load and preprocess images
2. Convert images into patches
3. Randomly mask 75% of patches
4. Encode visible patches with ViT encoder
5. Decode latent features to reconstruct masked patches
6. Compute reconstruction loss
7. Update model using AdamW optimizer

---

## Visualization

The project includes visualization tools to compare:

- Original image
- Masked image
- Reconstructed output

This helps evaluate how well the model learns semantic image structures.

---

## Evaluation Metrics

Two standard image reconstruction metrics are used:

### PSNR (Peak Signal-to-Noise Ratio)
Measures reconstruction fidelity.

### SSIM (Structural Similarity Index)
Measures structural similarity between original and reconstructed images.

---

## Gradio Web Application

The notebook provides a **Gradio interface** for testing the trained model.

Users can:

- Upload an image
- Apply masking
- View reconstructed results instantly

---

## Installation

```bash
pip install torch torchvision
pip install gradio
pip install scikit-image
pip install matplotlib numpy pillow
```

---

## Usage

Run the training notebook:

```bash
python train_mae.py
```

Launch the inference UI:

```bash
python app.py
```

Then open the Gradio interface in your browser.

---

## Applications

Masked Autoencoders can be used for:

- Self-supervised image representation learning
- Image reconstruction
- Computer vision pretraining
- Medical imaging
- Satellite imagery analysis
- Data-efficient deep learning

---

## Future Improvements

- Train on larger datasets (ImageNet-1K)
- Add distributed training
- Improve decoder architecture
- Extend to video masked autoencoders
- Deploy using Hugging Face Spaces

---

## References

- He et al. (2022) — Masked Autoencoders Are Scalable Vision Learners
- Vision Transformer (ViT)
- PyTorch Deep Learning Framework

---

## Author

Developed as a deep learning research project implementing **Masked Autoencoders with Vision Transformers for self-supervised image learning**.
