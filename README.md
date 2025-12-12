# Infrared Image Colorization

Thermal (IR) to RGB Image Colorization with GANs
This repository implements a Deep Learning based Colorization pipeline that converts single-channel Infrared (LWIR) images into 3-channel Visible (RGB) images using a Generative Adversarial Network (GAN).

The model is built with PyTorch and is specifically designed to work with the KAIST Multispectral Pedestrian Dataset. It uses a ResNet-based Generator and a PatchGAN Discriminator, utilizing Pix2Pix-style training (GAN Loss + L1 Loss).

📌 Features
Architecture: ResNet-9 Generator (with optional anti-aliasing) & PatchGAN Discriminator.

Loss Function: LSGAN (Least Squares GAN) + L1 Reconstruction Loss.

Normalization: Instance Normalization for better style transfer quality.

Data Handling: Custom DataLoader for the KAIST dataset (paired IR/RGB).

Modes: Supports both train (learning from scratch) and test (inference on folder).

## 1. Dataset: KAIST Multispectral

This project relies on the KAIST Multispectral Pedestrian Dataset:

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
        V000/
          lwir/
          visible/
        ...
```

In the current configuration:

set00 is used for training + validation.

set01/V000/lwir is used only as a test/inference folder.

You can easily change these in the Config class.

💡 The loader (KAISTPairDataset) only needs a root folder whose subfolders are Vxxx with lwir and visible inside.
For example, a typical training root can be:
kaist-dataset/versions/1/set00

## 2. KAIST MultispectralConfiguration:

All settings are managed inside the Config class at the top of the script. You don't need to pass command-line arguments; simply edit the script in your IDE (VS Code, PyCharm, etc.).

```text
class Config:
    def __init__(self):
        # MODE: "train" to train model, "test" to colorize images
        self.mode = "train" 
        
        # SYSTEM
        self.device = "cuda" 
        self.img_size = 256
        
        # PATHS
        self.kaist_root = r"path/to/kaist/set00"  # For Training
        self.save_dir = "./checkpoints_kaist"
        
        # HYPERPARAMETERS
        self.batch_size = 4
        self.epochs = 50
        self.lr_G = 2e-4
        ...
```

## 3. Usage

### 1. Training
Open the script.

Set self.mode = "train" in the Config class.

Set self.kaist_root to your dataset folder.

Run the script in Bash:

```text
python main.py
```

Checkpoints will be saved in ./checkpoints_kaist.

### 2. Testing (Inference)
Open the script.

Set self.mode = "test" in the Config class.

Set self.input_dir to the folder containing IR images you want to colorize.

Set self.test_G_weights to your trained model path (e.g., netG_epoch_050.pth).

Run the script in Bash:

```text
python main.py
```

Results will be saved in ./results.