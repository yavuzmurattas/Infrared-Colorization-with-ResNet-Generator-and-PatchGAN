## License / Usage
Copyright (c) 2026 Yavuz Murat TAŞ. All rights reserved.

No permission is granted to use, copy, modify, publish, distribute, sublicense,
or sell this software or its derivatives without explicit written permission.

# Color Anything

This repository implements an end-to-end **thermal / infrared (LWIR) → visible (RGB)** image colorization pipeline using a **Pix2Pix-style conditional GAN** in **PyTorch**.  
It is built to work with the **KAIST Multispectral Pedestrian Dataset** (paired `lwir/` and `visible/` frames).

The script supports two modes:

- **`train`**: train a conditional GAN on paired LWIR/RGB images
- **`test`**: run inference on KAIST test sets, save outputs, compute metrics (if GT exists), optionally save collages, and export Top-K best results

---
---

## 📊 Metrics & Results

Across the KAIST test splits, **Color Anything** provides the most consistent overall performance among the compared methods. In terms of average metrics (Table II), it yields the lowest reconstruction errors (**MAE/MSE**) and the highest perceptual quality (**PSNR/SSIM**), improving PSNR by **+0.17 dB** over **TIC-CGAN** and **+0.66 dB** over the **Naïve** baseline while also increasing SSIM. The best-case results (Table III) further highlight this gap, where Color Anything reaches **29.24 dB PSNR** and **0.89 SSIM**, indicating substantially improved colorization fidelity in favorable scenes. The ablation study (Table I) suggests that **training with both Day and Night data** generally improves robustness, with Night-only training being insufficient for daytime performance and Day-only training degrading night-time generalization. Table II and III are performed under day only training and day only testing. Best results are obtained in only day training condition by sacrificing the night cases.

When ground-truth exists (`visible/<frame>` is found), test mode computes:
- **MAE**
- **MSE**
- **PSNR**
- **SSIM** *(requires scikit-image; otherwise `None`)*

> Lower is better: **MAE, MSE**  
> Higher is better: **PSNR, SSIM**

# Table I: Summary of the ablation results

- **Train(DN)**: Train with Day and Night.
- **Train(D)**: Training with Day only.
- **Train(N)**: Training with Night only.
- **Test(DN)**: Test set with Day and Night.

| Training   | Test  | Count  | MAE    | MSE    | PSNR (dB) | SSIM |
|------------|-------|--------|--------|--------|-----------|------|
| Train(DN)  | DN    | 10,785 | 0.1552 | 0.0541 |   13.58   | 0.42 |
| Train(D)   | DN    | 10,785 | 0.1654 | 0.0609 |   13.07   | 0.41 |
| Train(N)   | DN    | 10,785 | 0.2226 | 0.0992 |   10.46   | 0.30 |
| Train(DN)  | Day   | 7,866  | 0.1459 | 0.0466 |   14.09   | 0.43 |
| Train(D)   | Day   | 7,866  | 0.1453 | 0.0467 |   14.10   | 0.43 |
| Train(N)   | Day   | 7,866  | 0.2391 | 0.1091 |   9.73    | 0.28 |
| Train(DN)  | Night | 2,919  | 0.1803 | 0.0742 |   12.22   | 0.37 |
| Train(D)   | Night | 2,919  | 0.2193 | 0.0993 |   10.28   | 0.34 |
| Train(N)   | Night | 2,919  | 0.1783 | 0.0728 |   12.41   | 0.36 |

# Table II: Average results of different methods

| Method         | MAE    | MSE    | PSNR  | SSIM |
|----------------|--------|--------|-------|------|
| Naïve          | 0.1541 | 0.0528 | 13.44 | 0.36 |
| TIC-CGAN       | 0.1547 | 0.0481 | 13.93 | 0.42 |
| Color Anything | 0.1453 | 0.0466 | 14.10 | 0.43 |

# Table III: Best results of different methods

| Method         | MAE    | MSE    | PSNR  | SSIM |
|----------------|--------|--------|-------|------|
| Naïve          | 0.0392 | 0.0037 | 24.21 | 0.70 |
| TIC-CGAN       | 0.0635 | 0.0100 | 19.93 | 0.65 |
| Color Anything | 0.0198 | 0.0010 | 29.24 | 0.89 |


## ✨ Highlights

- **Generator:** U-Net style encoder/decoder with **ResNet bottleneck (9 ResNet blocks)** + skip connections  
- **Discriminator:** **PatchGAN** (predicts a grid of real/fake scores per patch)
- **Losses (training):**
  - **Hinge GAN loss**
  - **L1 reconstruction loss**
  - **VGG-16 perceptual loss** (feature-space L1)
  - **Total Variation (TV)** regularization
  - **Differentiable SSIM loss** (implemented in PyTorch)
- **Optional anti-aliasing:**
  - `Downsample` (blur + stride) and `UpsampleAA` (bilinear upsample + blur)
- **Evaluation (test mode):**
  - MAE / MSE / PSNR
  - SSIM via **scikit-image** *(optional; evaluation metric only)*
- **Convenient outputs:**
  - Saves predictions in mirrored KAIST folder structure
  - Optional collages: **IR | Pred | GT**
  - Optional Top-K export into a single folder + ranking CSV

---

## 🧩 Project Assumptions

### Input / Output normalization
- IR input is treated as **grayscale** and normalized to **[-1, 1]**
- RGB output uses **tanh**, so predictions are in **[-1, 1]** and converted back to **[0, 255]** for saving

### Pairing rule
Pairs are created by matching filenames between:

- `.../<sequence>/lwir/<frame>.jpg`
- `.../<sequence>/visible/<frame>.jpg`

---

## 📦 Requirements

**Python 3.9+** recommended.

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy pillow
# optional (only for SSIM as a test metric):
pip install scikit-image
```

Notes:
- Training SSIM loss is implemented in PyTorch and does **not** require scikit-image.
- scikit-image is only used to compute SSIM as an **evaluation metric** in test mode.

---

## 🗂️ Dataset: KAIST Multispectral Pedestrian Dataset

Expected directory layout:

```text
kaist-dataset/
  versions/
    1/
      set00/
        V000/
          lwir/      # IR (thermal) frames
          visible/   # RGB frames
        V001/
          lwir/
          visible/
        ...
      set01/
      set02/
      set03/
      set04/
      set05/
      ...
```

The loader is **recursive**: it searches for `lwir` folders and checks for a sibling `visible` folder.

---

## ⚙️ Configuration

All configuration is defined in the `Config` class inside the script (no CLI args required).  
Edit the fields to match your environment.

Key fields:

- `mode`: `"train"` or `"test"`
- `device`: `"cuda"` or `"cpu"`
- `img_size`: resize resolution (default: 256)
- `train_roots`: KAIST sets used for training (default: set00, set01, set03, set04)
- `test_roots`: KAIST sets used for testing (default: set02, set05)
- `save_dir`: checkpoint directory
- `output_dir`: where predictions / collages / metrics are saved
- `test_G_weights`: checkpoint path used in test mode (e.g., `netG_best.pth`)

> ⚠️ Note  
> If `save_dir` is assigned more than once inside `Config`, the **last assignment wins**.

---

## 🚀 Usage

### 1) Train

1. In `Config`:
   - set `self.mode = "train"`
   - set `self.train_roots = [...]` to your KAIST training set directories
2. Run:

```bash
python main.py
```

What you get:
- Periodic checkpoints: `netG_epoch_XXX.pth` (every `save_every` epochs)
- Best checkpoint by validation L1: `netG_best.pth`

---

### 2) Test / Inference (+ Metrics)

1. In `Config`:
   - set `self.mode = "test"`
   - set `self.test_roots = [...]` (e.g., set02 & set05)
   - set `self.test_G_weights = "./checkpoints_kaist/netG_best.pth"`
2. Run:

```bash
python main.py
```

Predictions are saved to:

```text
<output_dir>/<set_name>/<sequence>/<filename>
```

Example:

```text
results/set02/V000/I00001.jpg
```

---

## 🖼️ Collages: IR | Pred | GT

If enabled:

```python
self.save_comparisons = True
self.comparison_dirname = "Comparisons"
```

Collages are saved to:

```text
<output_dir>/Comparisons/<set>/<sequence>/<stem>_cmp.png
```

### Disable all overlay text on collages
To remove **IR / Pred / GT labels** and **PSNR/SSIM text**, set:

```python
self.comparison_add_text = False
```

---

## 🏆 Top-K Export (Best Results)

If enabled:

```python
self.topk = 50
self.best50_dirname = "Best_50_colored_images"
self.best50_copy_preds = True
self.best50_copy_collages = True
```

Ranking rule:
- Uses **SSIM** if available (and computed), otherwise uses **PSNR**

Outputs:

```text
<output_dir>/Best_50_colored_images/colored/
<output_dir>/Best_50_colored_images/collages/
<output_dir>/Best_50_colored_images/top_50_ranking.csv
```

---

## 📊 Metrics

When ground-truth exists (`visible/<frame>` is found), test mode computes:

- **MAE**
- **MSE**
- **PSNR**
- **SSIM** *(requires scikit-image; otherwise `None`)*

A CSV is written to:

```text
<output_dir>/metrics_test.csv
```

---

## 🧠 Model Overview

### Generator (ResNet-U-Net)
- Initial 7×7 conv (ngf channels)
- Two downsampling stages (optional anti-aliasing)
- **9 ResNet blocks** at bottleneck
- Two upsampling stages (optional anti-aliasing)
- `tanh` output → RGB in **[-1, 1]**

### Discriminator (PatchGAN)
- Receives concatenated input during training:
  - IR (1 channel) + RGB (3 channels) → **4 channels**
- Outputs a patch score map (local realism)

---

## 🛠️ Troubleshooting

### SSIM is `None`
Install scikit-image:

```bash
pip install scikit-image
```

### “No IR-RGB pairs found…”
- Ensure each sequence contains **both** `lwir/` and `visible/`
- Ensure filenames match between LWIR and visible folders

### CUDA not used
Make sure PyTorch sees your GPU and `device` is set:

```python
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```
