import os
import random
from glob import glob
import math
import shutil

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import functools

# Perceptual loss backbone (VGG)
from torchvision import models

# SSIM metric (optional; used only during evaluation)
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


# =========================================================
# 0) Configuration
# =========================================================

class Config:
    """
    Central configuration container.

    This script supports two modes:
      - "train": Train a Pix2Pix-style cGAN (PatchGAN + LSGAN) using KAIST paired IR (LWIR) and visible RGB.
      - "test" : Run inference on KAIST test sets, save predictions, compute metrics (if GT exists),
                 optionally save side-by-side collages, and export Top-K best results.

    Notes about KAIST folder layout assumed by this code:
      - IR images live under:  <setXX>/<sequence>/lwir/
      - RGB images live under: <setXX>/<sequence>/visible/
      - Pairing is performed by matching filenames between lwir/ and visible/ directories.
    """
    def __init__(self):
        # "train" => train on KAIST IR→RGB pairs
        # "test"  => colorize IR images from test sets, compute metrics, and save outputs
        self.mode = "test"  # "train" or "test"

        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Target image resolution (all inputs/outputs are resized to img_size x img_size)
        self.img_size = 256

        # Channel counts
        self.input_nc = 1   # IR (grayscale)
        self.output_nc = 3  # RGB

        # Generator base feature width
        self.ngf = 64

        # Normalization type used in the generator and discriminator ("instance", "batch", or "none")
        self.norm = "instance"

        # Anti-aliasing controls for downsample/upsample operations
        self.no_antialias = False
        self.no_antialias_up = False

        self.save_dir = r".\Weights\trained_w_night\checkpoints_kaist"
        self.output_dir = r".\results"
        self.test_G_weights = r".\Weights\trained_w_night\checkpoints_kaist\netG_best.pth"

        # ---------- TRAIN (KAIST) ----------
        # Training sets: set00, set01, set03, set04
        self.train_roots = [
            r"kaist-dataset\versions\1\set00",
            r"kaist-dataset\versions\1\set01",
            r"kaist-dataset\versions\1\set03",
            r"kaist-dataset\versions\1\set04",
        ]

        # Kept for legacy logging; training actually uses train_roots
        self.kaist_root = self.train_roots[0]

        # Training hyperparameters
        self.batch_size = 4
        self.epochs = 50
        self.lr_G = 2e-4
        self.lr_D = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999

        # Reconstruction / perceptual / regularization loss weights
        self.lambda_L1 = 30.0          # pixel L1 term
        self.lambda_perc = 30.0         # VGG perceptual term
        self.lambda_tv = 1e-4           # total variation term
        self.lambda_ssim = 2.0
        self.lambda_gan = 0.1

        # DataLoader settings
        self.num_workers = 4

        # Checkpointing
        self.save_dir = "./checkpoints_kaist"
        self.save_every = 5

        # Validation split ratio (fraction of full training data)
        self.val_ratio = 0.1

        # Learning rate decay schedule:
        # Keep LR constant until lr_decay_start_epoch, then linearly decay to zero at the final epoch.
        self.lr_decay_start_epoch = 40

        # Optional: initialize generator weights from a checkpoint before training begins
        self.init_G_weights = None  # e.g., r"./pretrained_netG.pth"

        # ---------- TEST (INFERENCE + METRICS) ----------
        # Test sets: set02, set05
        self.test_roots = [
            r"kaist-dataset\versions\1\set02",
            r"kaist-dataset\versions\1\set05"
        ]

        # Save side-by-side comparison images (IR | Pred | GT if available)
        self.save_comparisons = True
        self.comparison_dirname = "Comparisons"
        self.comparison_add_text = False
        self.comparison_pad = 8
        self.comparison_font_scale = 0.6
        self.comparison_thickness = 2

        # Copy Top-K best results into a dedicated folder
        self.best50_copy_preds = True
        self.best50_copy_collages = True
        self.best50_preds_subdir = "colored"
        self.best50_collages_subdir = "collages"

        # Top-K selection size
        self.topk = 50
        self.best50_dirname = "Best_50_colored_images"

# =========================================================
# 1) Normalization and weight initialization helpers
# =========================================================

class Identity(nn.Module):
    """A pass-through layer used when normalization is disabled."""
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer constructor based on norm_type.
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    elif norm_type == 'none' or norm_type is None:
        return lambda num_features: Identity()
    else:
        raise NotImplementedError(f'Normalization type [{norm_type}] not supported')


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize weights for convolutional/linear layers, and normalization layers.

    - Conv/Linear: normal/xavier/kaiming/orthogonal initialization
    - Norm layers: weights ~ N(1, init_gain), bias = 0
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'init method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02,
             device=torch.device('cpu'), initialize_weights=True):
    """
    Move a network to the target device and optionally initialize its parameters.
    """
    net.to(device)
    if initialize_weights:
        init_weights(net, init_type, init_gain)
    return net


def get_lr_lambda(cfg: Config):
    """
    Build a scheduler function for linear learning-rate decay.

    The LambdaLR scheduler uses an epoch index starting from 0.
    This function maps that to an intuitive 1-based epoch number internally.
    """
    def lr_lambda(epoch):
        e = epoch + 1  # convert scheduler epoch (0-based) to training epoch (1-based)

        # Constant LR phase
        if e <= cfg.lr_decay_start_epoch:
            return 1.0
	else:
        # Decay phase
        	if e >= cfg.epochs:
            	return 0.0

        # Linearly decay from 1.0 to 0.0 between lr_decay_start_epoch and epochs
        frac = float(e - cfg.lr_decay_start_epoch) / float(max(1, cfg.epochs - cfg.lr_decay_start_epoch))
        return max(0.0, 1.0 - frac)

    return lr_lambda


# =========================================================
# 2) Anti-aliased downsample / upsample blocks
# =========================================================

def get_filter(filt_size=3):
    """
    Create a 2D binomial (approx. Gaussian) filter of size filt_size x filt_size.
    This is used to reduce aliasing when downsampling or to smooth after upsampling.

    filt_size must be in [1..7].
    """
    if filt_size == 1:
        a = np.array([1.], dtype=np.float32)
    elif filt_size == 2:
        a = np.array([1., 1.], dtype=np.float32)
    elif filt_size == 3:
        a = np.array([1., 2., 1.], dtype=np.float32)
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.], dtype=np.float32)
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.], dtype=np.float32)
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.], dtype=np.float32)
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.], dtype=np.float32)
    else:
        raise ValueError("filt_size must be 1-7")

    filt = a[:, None] * a[None, :]
    filt = filt / filt.sum()
    return torch.from_numpy(filt)


class Downsample(nn.Module):
    """
    Blur + strided convolution downsampling.

    Instead of using a strided conv directly (which can introduce aliasing),
    this performs reflection/replication/zero padding, then convolves with a fixed blur filter
    using stride > 1 to reduce spatial resolution.
    """
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.pad_off = pad_off

        pad = (filt_size - 1) / 2
        self.pad_sizes = [
            int(pad + pad_off),
            int(np.ceil(pad + pad_off)),
            int(pad + pad_off),
            int(np.ceil(pad + pad_off)),
        ]

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(self.pad_sizes)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(self.pad_sizes)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(self.pad_sizes)
        else:
            raise NotImplementedError

        filt = get_filter(filt_size)
        self.register_buffer(
            'filt',
            filt[None, None, :, :].repeat(channels, 1, 1, 1)
        )
        self.channels = channels

    def forward(self, x):
        x = self.pad(x)
        x = F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)
        return x


class UpsampleAA(nn.Module):
    """
    Anti-aliased upsampling.

    Approach:
      1) Bilinear resize to increase spatial resolution.
      2) Blur with a fixed low-pass filter to reduce ringing/aliasing artifacts.
    """
    def __init__(self, channels, filt_size=3, stride=2, pad_type='reflect'):
        super().__init__()
        self.stride = stride

        filt = get_filter(filt_size)
        self.register_buffer(
            'filt',
            filt[None, None, :, :].repeat(channels, 1, 1, 1)
        )

        pad = (filt_size - 1) / 2
        self.pad_sizes = [
            int(pad),
            int(np.ceil(pad)),
            int(pad),
            int(np.ceil(pad)),
        ]

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(self.pad_sizes)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(self.pad_sizes)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(self.pad_sizes)
        else:
            raise NotImplementedError

        self.channels = channels

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.stride,
                          mode='bilinear', align_corners=True)
        x = self.pad(x)
        x = F.conv2d(x, self.filt, stride=1, groups=self.channels)
        return x


# =========================================================
# 3) ResNet block (used in generator bottleneck)
# =========================================================

class ResnetBlock(nn.Module):
    """
    Standard ResNet block:
      - Conv -> Norm -> ReLU -> (optional Dropout) -> Conv -> Norm
      - Residual connection: output = input + block(input)
    """
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, use_bias=False):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        # First conv padding strategy
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'Padding [{padding_type}] is not implemented')

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        # Optional dropout helps regularization in some setups
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        # Second conv padding strategy
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'Padding [{padding_type}] is not implemented')

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# =========================================================
# 4) Generator: U-Net-like encoder/decoder with ResNet bottleneck
# =========================================================

class ResnetUNetGenerator(nn.Module):
    """
    A U-Net style generator with a ResNet bottleneck.

    High-level structure:
      - Encoder:
          c7s1-64
          down: 64 -> 128 (spatial /2)
          down: 128 -> 256 (spatial /2)
      - Bottleneck:
          N ResNet blocks at 256 channels
      - Decoder:
          up: 256 -> (concat skip 128) -> 128
          up: 128 -> (concat skip 64)  -> 64
          output: c7s1-3 + tanh

    Output is in [-1, 1] due to tanh activation.
    """
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect',
                 no_antialias=False, no_antialias_up=False):
        super().__init__()
        assert n_blocks >= 0

        # Bias usage depends on normalization choice
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        # Initial conv: 7x7, stride 1
        self.inc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        # Downsampling stage 1 (64 -> 128)
        # If no_antialias is False, we do conv stride 1 + blur-downsample.
        # If True, we do a stride-2 conv directly.
        stride_d = 1 if not no_antialias else 2
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=stride_d, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )
        self.down1_down = None if no_antialias else Downsample(ngf * 2)

        # Downsampling stage 2 (128 -> 256)
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=stride_d, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )
        self.down2_down = None if no_antialias else Downsample(ngf * 4)

        # Bottleneck: stack of ResNet blocks at 256 channels
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout, use_bias)
            )
        self.resblocks = nn.Sequential(*blocks)

        # Upsampling stage 1:
        # Upsample 256 channels to match x1 resolution, then concatenate skip from x1 (128),
        # then reduce to 128 channels.
        if no_antialias_up:
            self.up1_up = nn.ConvTranspose2d(
                ngf * 4, ngf * 4,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
            )
        else:
            self.up1_up = UpsampleAA(ngf * 4)

        self.up1_conv = nn.Sequential(
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )

        # Upsampling stage 2:
        # Upsample 128 channels to match x0 resolution, concatenate skip from x0 (64),
        # then reduce to 64 channels.
        if no_antialias_up:
            self.up2_up = nn.ConvTranspose2d(
                ngf * 2, ngf * 2,
                kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias
            )
        else:
            self.up2_up = UpsampleAA(ngf * 2)

        self.up2_conv = nn.Sequential(
            nn.Conv2d(ngf * 2 + ngf, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        # Final mapping to RGB with tanh to produce [-1, 1] range
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x, layers=None, encode_only=False):
        """
        Forward pass.

        The layers/encode_only parameters are included to keep the call signature compatible
        with some CUT-style code patterns, but they are not used here.
        """
        # Encoder
        x0 = self.inc(x)           # (B, 64, H, W)
        x1 = self.down1(x0)        # (B, 128, H/2, W/2) if stride-2, otherwise same
        if self.down1_down is not None:
            x1 = self.down1_down(x1)  # (B, 128, H/2, W/2)

        x2 = self.down2(x1)        # (B, 256, H/4, W/4)
        if self.down2_down is not None:
            x2 = self.down2_down(x2)  # (B, 256, H/4, W/4)

        # Bottleneck
        x3 = self.resblocks(x2)    # (B, 256, H/4, W/4)

        # Decoder stage 1 (skip from x1)
        y = self.up1_up(x3)        # (B, 256, approx H/2, approx W/2)
        if y.shape[-2:] != x1.shape[-2:]:
            y = F.interpolate(y, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, x1], dim=1)   # (B, 256+128, H/2, W/2)
        y = self.up1_conv(y)            # (B, 128, H/2, W/2)

        # Decoder stage 2 (skip from x0)
        y = self.up2_up(y)              # (B, 128, approx H, approx W)
        if y.shape[-2:] != x0.shape[-2:]:
            y = F.interpolate(y, size=x0.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, x0], dim=1)   # (B, 128+64, H, W)
        y = self.up2_conv(y)            # (B, 64, H, W)

        # Output
        out = self.outc(y)              # (B, 3, H, W) in [-1, 1]
        return out, None


# =========================================================
# 5) Discriminator: PatchGAN
# =========================================================

class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    It predicts a grid of real/fake scores (one score per patch) rather than a single scalar.
    This helps enforce local realism and sharpness.

    Input: concatenation of IR (1ch) and RGB (3ch) => total 4 channels during training.
    """
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()

        # Bias usage depends on normalization choice
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4
        padw = 1

        # First layer: conv + LeakyReLU (no normalization)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        # Middle layers: progressively increase channel depth
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # Penultimate layer: stride 1 for finer patch resolution
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Final layer: output 1-channel patch score map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# =========================================================
# 6) Perceptual loss (VGG-16) + Total Variation loss
# =========================================================

class VGGPerceptual(nn.Module):
    """
    Perceptual feature extractor using VGG-16 pretrained on ImageNet.

    Expected input:
      - x is Bx3xHxW in [-1, 1]
    Processing:
      - Convert to [0, 1]
      - Apply ImageNet mean/std normalization
      - Extract early/mid-level features

    This module is used to compute an L1 distance in feature space between prediction and target.
    """
    def __init__(self, device):
        super().__init__()
        # Torchvision weights API (with fallback for older versions)
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:
            vgg = models.vgg16(pretrained=True)

        # Use features up to relu3_3 (approximately features[:16])
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).to(device)

        # Freeze parameters (no gradient updates)
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        # ImageNet normalization buffers (broadcastable)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # x: Bx3xHxW in [-1,1] -> [0,1]
        x = (x + 1.0) / 2.0

        # Normalize for VGG
        x = (x - self.mean) / self.std
        return self.features(x)


def tv_loss(x):
    """
    Total variation loss encourages spatial smoothness in the output.

    It penalizes absolute differences between neighboring pixels horizontally and vertically.
    """
    diff_i = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    diff_j = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return diff_i + diff_j


# ---------- Differentiable SSIM in PyTorch (for training loss) ----------

def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_ssim_window(window_size, channel, device, dtype):
    _1d = _gaussian_window(window_size, sigma=1.5, device=device, dtype=dtype).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_loss_torch(img1, img2, window_size=11, size_average=True):
    """
    img1, img2: tensors in [0,1], shape BxCxHxW
    Returns: 1 - SSIM
    """
    assert img1.shape == img2.shape, "SSIM images must have the same shape"
    b, c, h, w = img1.shape

    device = img1.device
    dtype = img1.dtype

    window = _create_ssim_window(window_size, c, device, dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=c)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=c) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        ssim_val = ssim_map.mean()
    else:
        ssim_val = ssim_map.mean(dim=[1, 2, 3])  # B

    # Loss: 1 - SSIM (SSIM in [0,1])
    return 1.0 - ssim_val


# =========================================================
# 7) Model wrapper (generator only for inference; generator+discriminator for training)
# =========================================================

class IRColorizationModel(nn.Module):
    """
    Lightweight wrapper that holds the generator network and provides:
      - weight loading
      - forward method for IR -> RGB
    """
    def __init__(self, cfg: Config):
        super().__init__()
        norm_layer = get_norm_layer(cfg.norm)

        self.netG = ResnetUNetGenerator(
            cfg.input_nc, cfg.output_nc, cfg.ngf,
            norm_layer=norm_layer,
            use_dropout=False,
            n_blocks=9,
            padding_type='reflect',
            no_antialias=cfg.no_antialias,
            no_antialias_up=cfg.no_antialias_up,
        )

        self.device = torch.device(cfg.device)
        self.netG = init_net(self.netG, init_type='normal', init_gain=0.02,
                             device=self.device, initialize_weights=True)

    def load_weights(self, path):
        """
        Load generator weights from a checkpoint.
        Supports either a raw state_dict or a dict containing 'state_dict'.
        """
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.netG.load_state_dict(state, strict=False)

    def forward(self, ir_tensor):
        """
        Forward IR tensor (Bx1xHxW in [-1,1]) through generator to produce RGB (Bx3xHxW in [-1,1]).
        """
        fake_b, _ = self.netG(ir_tensor)
        return fake_b


# =========================================================
# 8) Inference utilities (image I/O, tensor conversion)
# =========================================================

def load_ir_image(path, img_size=None):
    """
    Load a grayscale IR image from disk.

    Returns:
      - HxW float32 in [0, 1]

    If img_size is provided, the image is resized to (img_size, img_size).
    """
    img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_u8 is None:
        raise RuntimeError(f"Could not read image: {path}")
    orig_dtype = img_u8.dtype

    if img_size is not None:
        img_u8 = cv2.resize(img_u8, (img_size, img_size), interpolation=cv2.INTER_AREA)

    img = img_u8.astype(np.float32)

    # Handle both 8-bit and 16-bit sources
    if img.max() > 1.0:
        if orig_dtype == np.uint8:
            img /= 255.0
        else:
            img /= 65535.0

    img = np.clip(img, 0.0, 1.0)
    return img


def load_rgb_image(path, img_size=None):
    """
    Load a color (RGB) image from disk.

    Returns:
      - HxWx3 float32 in [0, 1]

    If img_size is provided, the image is resized to (img_size, img_size).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read RGB image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img_size is not None:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def ir_to_tensor(img_hw):
    """
    Convert a grayscale IR image in [0,1] (HxW) to a torch tensor in [-1,1] (1x1xHxW).
    """
    img = img_hw[None, None, :, :]
    img = torch.from_numpy(img).float()
    img = img * 2.0 - 1.0
    return img


def tensor_to_rgb_image(tensor_bchw):
    """
    Convert a generated RGB tensor in [-1,1] (Bx3xHxW) to a uint8 image (HxWx3) in [0,255].

    Only the first element of the batch is converted.
    """
    x = tensor_bchw[0].detach().cpu().numpy()
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return x


def save_rgb(path, img_rgb):
    """
    Save an RGB uint8 image (HxWx3) to disk using PIL.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_rgb).save(path)


def collect_images(input_dir):
    """
    Collect image file paths directly under input_dir (non-recursive).
    """
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_dir, ext)))
    return sorted(files)


def collect_kaist_ir_files_from_sets(set_roots):
    """
    Recursively scan KAIST set directories and collect IR files under any 'lwir' folder.

    For each discovered IR file, record:
      - ir_path: absolute path to the IR image
      - set_name: the set folder name (e.g., set02)
      - seq_rel : the relative sequence path from the set root (e.g., V000 or deeper)

    Only 'lwir' folders that have a sibling 'visible' folder are considered valid.
    """
    if isinstance(set_roots, (str, bytes)):
        set_roots = [set_roots]

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    entries = []

    def list_imgs(folder):
        if not os.path.isdir(folder):
            return []
        files = []
        for fn in os.listdir(folder):
            if fn.lower().endswith(exts):
                files.append(os.path.join(folder, fn))
        return sorted(files)

    for root in set_roots:
        if not os.path.isdir(root):
            print(f"[WARN] set root not found: {root}")
            continue

        set_name = os.path.basename(root.rstrip("\\/"))

        # Walk the directory tree to find any folder named 'lwir'
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath).lower() != "lwir":
                continue

            lwir_dir = dirpath
            seq_dir = os.path.dirname(lwir_dir)        # .../<sequence>
            vis_dir = os.path.join(seq_dir, "visible") # .../<sequence>/visible

            if not os.path.isdir(vis_dir):
                continue

            ir_files = list_imgs(lwir_dir)
            if len(ir_files) == 0:
                continue

            seq_rel = os.path.relpath(seq_dir, root)

            for ir_path in ir_files:
                base = os.path.basename(ir_path)
                entries.append((ir_path, set_name, seq_rel))

    return entries


def float01_to_uint8_rgb(img01_hw_or_hwc):
    """
    Convert:
      - HxW float [0,1] (grayscale) or
      - HxWx3 float [0,1] (RGB)
    into HxWx3 uint8 [0,255].

    If input is grayscale, it is replicated into 3 channels.
    """
    x = np.clip(img01_hw_or_hwc, 0.0, 1.0)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=2)
    x = (x * 255.0).astype(np.uint8)
    return x


def make_comparison_collage(ir01_hw, pred_u8_hwc, gt01_hwc=None,
                            add_text=True, pad=8,
                            font_scale=0.6, thickness=2,
                            metrics_text=None):
    """
    Create a side-by-side collage:

      [ IR (gray replicated to RGB) | Pred (RGB) | GT (RGB, optional) ]

    Parameters:
      - ir01_hw      : HxW float [0,1]
      - pred_u8_hwc  : HxWx3 uint8
      - gt01_hwc     : HxWx3 float [0,1] or None
      - metrics_text : optional string to overlay (e.g., "PSNR=23.1dB SSIM=0.71")

    Returns:
      - collage image as HxWx3 uint8
    """
    ir_u8 = float01_to_uint8_rgb(ir01_hw)
    pred = pred_u8_hwc
    imgs = [ir_u8, pred]

    if gt01_hwc is not None:
        gt_u8 = float01_to_uint8_rgb(gt01_hwc)
        imgs.append(gt_u8)

    H = imgs[0].shape[0]
    widths = [im.shape[1] for im in imgs]
    total_w = sum(widths) + pad * (len(imgs) - 1)

    canvas = np.zeros((H, total_w, 3), dtype=np.uint8)

    x = 0
    for k, im in enumerate(imgs):
        canvas[:, x:x + im.shape[1], :] = im
        x += im.shape[1]
        if k != len(imgs) - 1:
            x += pad

    if add_text:
        # cv2 uses BGR for colors, but on RGB canvas text is still readable.
        cv2.putText(canvas, "IR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        x_pred = widths[0] + pad + 10
        cv2.putText(canvas, "Pred", (x_pred, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if gt01_hwc is not None:
            x_gt = widths[0] + pad + widths[1] + pad + 10
            cv2.putText(canvas, "GT", (x_gt, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if metrics_text is not None:
            cv2.putText(canvas, metrics_text, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas


def save_comparison_image(cfg, out_rel, collage_u8_hwc):
    """
    Save collage under:
      <output_dir>/<comparison_dirname>/<subdirs>/<stem>_cmp.png

    out_rel is the prediction's relative path, such as:
      set02/V000/I00001.jpg
    """
    base = os.path.basename(out_rel)
    stem, _ = os.path.splitext(base)
    subdir = os.path.dirname(out_rel)

    cmp_dir = os.path.join(cfg.output_dir, cfg.comparison_dirname, subdir)
    os.makedirs(cmp_dir, exist_ok=True)

    cmp_path = os.path.join(cmp_dir, f"{stem}_cmp.png")
    Image.fromarray(collage_u8_hwc).save(cmp_path)
    return cmp_path


# =========================================================
# 9) KAIST Dataset (Vxxx/lwir, Vxxx/visible)
# =========================================================

class KAISTPairDataset(Dataset):
    """
    KAIST IR-RGB paired dataset loader (recursive).

    It scans each root directory and finds folders named:
      - lwir     (IR images)
      - visible  (RGB images)

    Pairing logic:
      - Within each (sequence) folder, it matches IR and RGB images by filename intersection.
      - Each sample yields:
          {'ir': 1xHxW tensor in [-1,1],
           'rgb': 3xHxW tensor in [-1,1]}

    Augmentations:
      - Random horizontal flip (applied consistently to both IR and RGB).
    """
    def __init__(self, root, img_size=256, augment=True, indices=None):
        super().__init__()
        self.img_size = img_size
        self.augment = augment

        if isinstance(root, (list, tuple)):
            roots = list(root)
        else:
            roots = [root]

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_ir = []
        all_rgb = []

        def list_imgs_map(folder):
            """Return dict: filename -> full path for image files under folder."""
            m = {}
            if not os.path.isdir(folder):
                return m
            for fn in os.listdir(folder):
                if fn.lower().endswith(exts):
                    m[fn] = os.path.join(folder, fn)
            return m

        def scan_one_root(one_root):
            if not os.path.isdir(one_root):
                return

            for dirpath, dirnames, filenames in os.walk(one_root):
                if os.path.basename(dirpath).lower() != "lwir":
                    continue

                lwir_dir = dirpath
                seq_dir = os.path.dirname(lwir_dir)
                vis_dir = os.path.join(seq_dir, "visible")
                if not os.path.isdir(vis_dir):
                    continue

                ir_map = list_imgs_map(lwir_dir)
                rgb_map = list_imgs_map(vis_dir)
                if len(ir_map) == 0 or len(rgb_map) == 0:
                    continue

                common = sorted(set(ir_map.keys()) & set(rgb_map.keys()))
                if len(common) == 0:
                    continue

                for fn in common:
                    all_ir.append(ir_map[fn])
                    all_rgb.append(rgb_map[fn])

        for r in roots:
            scan_one_root(r)

        if len(all_ir) == 0:
            raise RuntimeError(f"No IR-RGB pairs found under roots: {roots}")

        # Apply an index subset if provided (used for train/val split)
        if indices is not None:
            self.ir_paths = [all_ir[i] for i in indices]
            self.rgb_paths = [all_rgb[i] for i in indices]
        else:
            self.ir_paths = all_ir
            self.rgb_paths = all_rgb

        print(f"[KAISTPairDataset] total pairs: {len(self.ir_paths)} (augment={self.augment})")

    def __len__(self):
        return len(self.ir_paths)

    def _read_ir(self, path):
        """Read IR grayscale image, resize, and return float [0,1]."""
        img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_u8 is None:
            raise RuntimeError(f"Could not read IR image: {path}")

        orig_dtype = img_u8.dtype
        img_u8 = cv2.resize(img_u8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img = img_u8.astype(np.float32)
        if img.max() > 1.0:
            if orig_dtype == np.uint8:
                img /= 255.0
            else:
                img /= 65535.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def _read_rgb(self, path):
        """Read RGB image, resize, and return float [0,1] with RGB channel order."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read RGB image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def __getitem__(self, idx):
        ir = self._read_ir(self.ir_paths[idx])
        rgb = self._read_rgb(self.rgb_paths[idx])

        # Simple paired augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            ir = np.fliplr(ir).copy()
            rgb = np.fliplr(rgb).copy()

        # Convert to tensors
        ir_t = torch.from_numpy(ir).unsqueeze(0)                 # 1 x H x W
        rgb_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))   # 3 x H x W

        # Normalize to [-1,1] for tanh-based generator
        ir_t = ir_t * 2.0 - 1.0
        rgb_t = rgb_t * 2.0 - 1.0
        return {'ir': ir_t, 'rgb': rgb_t}


# =========================================================
# 10) Evaluation: inference + metrics + saving
# =========================================================

def compute_metrics(pred_01, gt_01):
    """
    Compute basic image quality metrics between prediction and ground-truth.

    Inputs:
      - pred_01, gt_01: HxWx3 float32 images in [0,1]

    Returns:
      - mae : mean absolute error
      - mse : mean squared error
      - psnr: peak signal-to-noise ratio (computed from MSE assuming peak=1.0)
      - ssim_val: structural similarity index (None if scikit-image is unavailable)
    """
    diff = pred_01 - gt_01
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    # PSNR definition using peak=1.0
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20.0 * math.log10(1.0) - 10.0 * math.log10(mse + 1e-12)

    # SSIM (optional)
    if HAVE_SKIMAGE:
        try:
            ssim_val = float(ssim(gt_01, pred_01, data_range=1.0, channel_axis=2))
        except TypeError:
            # Older scikit-image versions use multichannel=True
            ssim_val = float(ssim(gt_01, pred_01, data_range=1.0, multichannel=True))
    else:
        ssim_val = None

    return mae, mse, psnr, ssim_val


def save_best_k_outputs(cfg: Config, metrics_list):
    """
    Copy the top-K outputs (best quality) into a dedicated folder.

    Ranking metric:
      - Use SSIM if available and computed
      - Otherwise use PSNR

    Files copied:
      - Predictions (colored images) into: <output_dir>/<best50_dirname>/<colored_subdir>/
      - Collages into: <output_dir>/<best50_dirname>/<collages_subdir>/

    A ranking CSV is also written to document the ordering and metric values.
    """
    if not metrics_list:
        print("[TOP-K] metrics_list empty, skipping top-K save.")
        return

    if HAVE_SKIMAGE and any(m.get("ssim") is not None for m in metrics_list):
        metric_key = "ssim"
    else:
        metric_key = "psnr"

    # Filter invalid entries for the selected metric
    valid = []
    for m in metrics_list:
        v = m.get(metric_key, None)
        if v is None:
            continue
        if isinstance(v, float) and (not np.isfinite(v)):
            continue
        valid.append(m)

    if not valid:
        print(f"[TOP-K] No valid '{metric_key}' values, skipping top-K save.")
        return

    # Sort descending: higher is better
    valid.sort(key=lambda x: x[metric_key], reverse=True)
    top_k = valid[:max(1, int(cfg.topk))]

    best_dir = os.path.join(cfg.output_dir, cfg.best50_dirname)
    os.makedirs(best_dir, exist_ok=True)

    preds_sub = os.path.join(best_dir, getattr(cfg, "best50_preds_subdir", "colored"))
    colls_sub = os.path.join(best_dir, getattr(cfg, "best50_collages_subdir", "collages"))
    os.makedirs(preds_sub, exist_ok=True)
    os.makedirs(colls_sub, exist_ok=True)

    # Ranking CSV
    rank_path = os.path.join(best_dir, f"top_{len(top_k)}_ranking.csv")
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write("rank,file,mae,mse,psnr,ssim,metric_used\n")
        for r, m in enumerate(top_k, start=1):
            ssim_val = m.get("ssim", None)
            ssim_str = "" if ssim_val is None else f"{ssim_val:.6f}"
            f.write(
                f"{r},{m['file']},{m['mae']:.8f},{m['mse']:.8f},{m['psnr']:.6f},{ssim_str},{metric_key}\n"
            )

    copy_preds = getattr(cfg, "best50_copy_preds", True)
    copy_colls = getattr(cfg, "best50_copy_collages", True)

    copied_preds = 0
    copied_colls = 0

    for m in top_k:
        rel_id = m["file"]  # e.g., set02/V000/I00001.jpg
        rel_norm = rel_id.replace("\\", "/")
        subdir = os.path.dirname(rel_norm)          # e.g., set02/V000
        base = os.path.basename(rel_norm)           # e.g., I00001.jpg
        stem, _ = os.path.splitext(base)

        # Flatten name to avoid collisions across folders
        flat_base = rel_norm.replace("/", "__")     # e.g., set02__V000__I00001.jpg
        flat_stem = os.path.splitext(flat_base)[0]  # e.g., set02__V000__I00001

        # 1) Copy prediction image
        if copy_preds:
            src_pred = os.path.join(cfg.output_dir, rel_id)
            dst_pred = os.path.join(preds_sub, flat_base)
            if os.path.isfile(src_pred):
                shutil.copy2(src_pred, dst_pred)
                copied_preds += 1
            else:
                print(f"[TOP-K][WARN] Missing prediction, cannot copy: {src_pred}")

        # 2) Copy collage image
        if copy_colls:
            # Collage path convention:
            #   <output_dir>/<comparison_dirname>/<set>/<seq>/<stem>_cmp.png
            src_cmp = os.path.join(cfg.output_dir, cfg.comparison_dirname, subdir, f"{stem}_cmp.png")

            # If saved with a different extension, try jpg
            if not os.path.isfile(src_cmp):
                src_cmp_jpg = os.path.join(cfg.output_dir, cfg.comparison_dirname, subdir, f"{stem}_cmp.jpg")
                if os.path.isfile(src_cmp_jpg):
                    src_cmp = src_cmp_jpg

            dst_cmp = os.path.join(colls_sub, f"{flat_stem}__cmp.png")

            if os.path.isfile(src_cmp):
                shutil.copy2(src_cmp, dst_cmp)
                copied_colls += 1
            else:
                print(f"[TOP-K][WARN] Missing collage, cannot copy: {src_cmp}")

    print(f"[TOP-K] Saved best outputs to: {best_dir}")
    print(f"[TOP-K] Colored copied : {copied_preds}/{len(top_k)} -> {preds_sub}")
    print(f"[TOP-K] Collage copied : {copied_colls}/{len(top_k)} -> {colls_sub}")
    print(f"[TOP-K] Ranking file   : {rank_path}")


def run_test(cfg: Config):
    """
    Inference runner:
      - Loads generator weights (if available)
      - Scans KAIST test sets (cfg.test_roots)
      - Saves prediction images in a mirrored folder structure
      - Computes metrics where GT exists
      - Saves collages and top-K best results
    """
    device = torch.device(cfg.device)
    print(f"[TEST] Device: {device}")

    if not HAVE_SKIMAGE:
        print("WARNING: scikit-image is not installed. SSIM will be reported as None.")
        print("Install via: pip install scikit-image")

    model = IRColorizationModel(cfg)
    if cfg.test_G_weights is not None and os.path.isfile(cfg.test_G_weights):
        print(f"Loading generator weights from: {cfg.test_G_weights}")
        model.load_weights(cfg.test_G_weights)
    else:
        print("WARNING: cfg.test_G_weights is None or does not exist; "
              "generator is randomly initialized, results will be meaningless.")

    model.eval()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Prefer scanning test_roots recursively; fallback to single input_dir if test_roots is empty
    if hasattr(cfg, "test_roots") and cfg.test_roots:
        entries = collect_kaist_ir_files_from_sets(cfg.test_roots)
        print(f"Found {len(entries)} IR images across test sets: {cfg.test_roots}")
    else:
        img_paths = collect_images(cfg.input_dir)
        entries = [(p, "input_dir", "seq") for p in img_paths]
        print(f"Found {len(entries)} IR images in {cfg.input_dir}")

    metrics_list = []
    sum_mae = 0.0
    sum_mse = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    count = 0

    best_psnr = -1.0
    best_psnr_sample = None
    best_ssim = -1.0
    best_ssim_sample = None

    for idx, (ir_path, set_name, seq_name) in enumerate(entries, start=1):
        # Load IR and convert to model tensor
        ir = load_ir_image(ir_path, img_size=cfg.img_size)
        ir_tensor = ir_to_tensor(ir).to(device)

        # Run generator
        with torch.no_grad():
            fake_rgb = model(ir_tensor)

        # Convert output to uint8 RGB image
        fake_rgb_np = tensor_to_rgb_image(fake_rgb)
        base = os.path.basename(ir_path)

        # Save prediction:
        #   <output_dir>/<set_name>/<seq_rel>/<filename>
        out_rel = os.path.join(set_name, seq_name, base)
        out_path = os.path.join(cfg.output_dir, out_rel)
        save_rgb(out_path, fake_rgb_np)

        # Locate GT RGB:
        # IR: .../<seq>/lwir/<file>
        # GT: .../<seq>/visible/<file>
        lwir_dir = os.path.dirname(ir_path)
        seq_dir = os.path.dirname(lwir_dir)
        vis_dir = os.path.join(seq_dir, "visible")
        gt_path = os.path.join(vis_dir, base)

        gt_rgb_01 = None
        mae = mse = psnr_val = None
        ssim_val = None

        # Compute metrics only if GT exists
        if os.path.isdir(vis_dir) and os.path.isfile(gt_path):
            gt_rgb_01 = load_rgb_image(gt_path, img_size=cfg.img_size)
            pred_rgb_01 = fake_rgb_np.astype(np.float32) / 255.0

            mae, mse, psnr_val, ssim_val = compute_metrics(pred_rgb_01, gt_rgb_01)

            metrics_list.append({
                "file": out_rel,
                "mae": mae,
                "mse": mse,
                "psnr": psnr_val,
                "ssim": ssim_val,
            })

            sum_mae += mae
            sum_mse += mse
            if np.isfinite(psnr_val):
                sum_psnr += psnr_val
            if ssim_val is not None:
                sum_ssim += ssim_val
            count += 1

            if np.isfinite(psnr_val) and psnr_val > best_psnr:
                best_psnr = psnr_val
                best_psnr_sample = out_rel
            if ssim_val is not None and ssim_val > best_ssim:
                best_ssim = ssim_val
                best_ssim_sample = out_rel
        else:
            if os.path.isdir(vis_dir):
                print(f"[WARN] No GT RGB found for {base} at {gt_path}; metrics skipped for this image.")

        # Save side-by-side collage if enabled
        if hasattr(cfg, "save_comparisons") and cfg.save_comparisons:
            metrics_text = None
            if (psnr_val is not None) and (ssim_val is not None):
                metrics_text = f"PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}"
            elif (psnr_val is not None):
                metrics_text = f"PSNR={psnr_val:.2f}dB"

            collage = make_comparison_collage(
                ir01_hw=ir,
                pred_u8_hwc=fake_rgb_np,
                gt01_hwc=gt_rgb_01,
                add_text=getattr(cfg, "comparison_add_text", True),
                pad=getattr(cfg, "comparison_pad", 8),
                font_scale=getattr(cfg, "comparison_font_scale", 0.6),
                thickness=getattr(cfg, "comparison_thickness", 2),
                metrics_text=metrics_text
            )
            cmp_path = save_comparison_image(cfg, out_rel, collage)

        # Periodic progress logging
        if idx % 50 == 0 or idx == len(entries):
            print(f"[{idx}/{len(entries)}] {ir_path} -> {out_path}")

    print("Test finished.")

    # Summarize metrics over all samples where GT was available
    if count > 0:
        mean_mae = sum_mae / count
        mean_mse = sum_mse / count
        mean_psnr = sum_psnr / count
        mean_ssim = (sum_ssim / count) if (HAVE_SKIMAGE and count > 0) else None

        print("\n=== Test Metrics (on images with GT) ===")
        print(f"Count      : {count}")
        print(f"Mean MAE   : {mean_mae:.6f}")
        print(f"Mean MSE   : {mean_mse:.6f}")
        print(f"Mean PSNR  : {mean_psnr:.4f} dB")
        if HAVE_SKIMAGE:
            print(f"Mean SSIM  : {mean_ssim:.6f}")
        else:
            print("Mean SSIM  : None (scikit-image not installed)")
        print(f"Best PSNR  : {best_psnr:.4f} ({best_psnr_sample})" if best_psnr_sample else "Best PSNR  : N/A")
        if HAVE_SKIMAGE and best_ssim_sample is not None:
            print(f"Best SSIM  : {best_ssim:.6f} ({best_ssim_sample})")
        else:
            print("Best SSIM  : N/A")

        # Write per-image metrics to CSV
        metrics_path = os.path.join(cfg.output_dir, "metrics_test.csv")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("file,mae,mse,psnr,ssim\n")
            for m in metrics_list:
                ssim_str = "" if m["ssim"] is None else f"{m['ssim']:.6f}"
                f.write(f"{m['file']},{m['mae']:.8f},{m['mse']:.8f},{m['psnr']:.6f},{ssim_str}\n")

            f.write("\n# Summary\n")
            f.write(f"# count,{count}\n")
            f.write(f"# mean_mae,{mean_mae:.8f}\n")
            f.write(f"# mean_mse,{mean_mse:.8f}\n")
            f.write(f"# mean_psnr,{mean_psnr:.6f}\n")
            if mean_ssim is not None:
                f.write(f"# mean_ssim,{mean_ssim:.6f}\n")
            else:
                f.write(f"# mean_ssim,\n")

        print(f"\nMetrics saved to: {metrics_path}")

        # Copy top-K results
        save_best_k_outputs(cfg, metrics_list)
    else:
        print("No metrics were computed (no matching GT RGB images found).")


# =========================================================
# 11) Validation helper (L1 only)
# =========================================================

def validate_kaist(model: IRColorizationModel, val_loader, device):
    """
    Compute validation loss using only pixel-wise L1 distance.

    This keeps validation fast and stable, while training may include GAN, perceptual, and TV terms.
    """
    model.eval()
    l1 = nn.L1Loss()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            ir = batch['ir'].to(device)
            rgb = batch['rgb'].to(device)
            fake = model(ir)
            loss = l1(fake, rgb)
            total_loss += loss.item() * ir.size(0)
            count += ir.size(0)

    model.train()
    return total_loss / max(count, 1)


# =========================================================
# 12) Train loop (Hinge GAN + L1 + Perceptual + TV + SSIM)
# =========================================================

def train_kaist(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[TRAIN] Device: {device}")
    print(f"KAIST root (V000, V001, ...): {cfg.kaist_root}")

    # Collect the full dataset first
    base_dataset = KAISTPairDataset(cfg.train_roots, img_size=cfg.img_size,
                                   augment=False, indices=None)

    N = len(base_dataset)
    val_size = max(1, int(N * cfg.val_ratio))
    train_size = N - val_size
    print(f"Total pairs: {N}, train: {train_size}, val: {val_size}")

    # Reproducible shuffle for splitting
    idxs = list(range(N))
    random.seed(42)
    random.shuffle(idxs)
    train_indices = idxs[:train_size]
    val_indices = idxs[train_size:]

    # Build train/val datasets referencing the same underlying (IR,RGB) pairing
    train_dataset = KAISTPairDataset(cfg.train_roots, img_size=cfg.img_size,
                                    augment=True, indices=train_indices)
    val_dataset = KAISTPairDataset(cfg.train_roots, img_size=cfg.img_size,
                                  augment=False, indices=val_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, drop_last=False)

    # Generator wrapper
    model = IRColorizationModel(cfg)
    if cfg.init_G_weights is not None and os.path.isfile(cfg.init_G_weights):
        print(f"Initializing generator from: {cfg.init_G_weights}")
        model.load_weights(cfg.init_G_weights)

    # Discriminator
    norm_layer = get_norm_layer(cfg.norm)
    netD = NLayerDiscriminator(
        input_nc=cfg.input_nc + cfg.output_nc,  # concat IR(1) + RGB(3)
        ndf=64,
        n_layers=3,
        norm_layer=norm_layer,
    )
    netD = init_net(netD, init_type='normal', init_gain=0.02,
                    device=device, initialize_weights=True)

    # Optimizers
    optimizerG = torch.optim.Adam(model.netG.parameters(),
                                  lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
    optimizerD = torch.optim.Adam(netD.parameters(),
                                  lr=cfg.lr_D, betas=(cfg.beta1, cfg.beta2))

    # LR schedulers (linear decay)
    lr_lambda = get_lr_lambda(cfg)
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_lambda)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda)

    criterionL1 = nn.L1Loss()

    # Perceptual loss model (VGG)
    vgg_perc = VGGPerceptual(device).to(device)

    os.makedirs(cfg.save_dir, exist_ok=True)

    best_val_l1 = float("inf")
    best_ckpt_path = os.path.join(cfg.save_dir, "netG_best.pth")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        netD.train()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        steps = 0

        for i, batch in enumerate(train_loader, start=1):
            ir = batch['ir'].to(device)
            rgb = batch['rgb'].to(device)

            # --------------------
            #  Update Discriminator D (Hinge loss)
            # --------------------
            optimizerD.zero_grad()
            with torch.no_grad():
                fake_rgb_detached = model(ir)
            real_input = torch.cat([ir, rgb], dim=1)
            fake_input = torch.cat([ir, fake_rgb_detached], dim=1)

            pred_real = netD(real_input)
            pred_fake = netD(fake_input)

            # Hinge loss for D:
            # L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
            loss_D_real = F.relu(1.0 - pred_real).mean()
            loss_D_fake = F.relu(1.0 + pred_fake).mean()
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizerD.step()

            # --------------------
            #  Update Generator G
            # --------------------
            optimizerG.zero_grad()
            fake_rgb = model(ir)
            fake_input = torch.cat([ir, fake_rgb], dim=1)
            pred_fake_for_G = netD(fake_input)

            # Hinge loss for G: L_G = -E[D(fake)]
            loss_G_GAN = -pred_fake_for_G.mean()

            loss_G_L1 = criterionL1(fake_rgb, rgb) * cfg.lambda_L1

            # Perceptual loss
            feat_fake = vgg_perc(fake_rgb)
            feat_real = vgg_perc(rgb)
            loss_G_perc = F.l1_loss(feat_fake, feat_real) * cfg.lambda_perc

            # TV loss
            loss_G_TV = tv_loss(fake_rgb) * cfg.lambda_tv

            # SSIM loss (images to [0,1] for structural similarity)
            fake_01 = (fake_rgb + 1.0) / 2.0
            real_01 = (rgb + 1.0) / 2.0
            loss_G_ssim = ssim_loss_torch(fake_01, real_01) * cfg.lambda_ssim

            loss_G = cfg.lambda_gan*loss_G_GAN + loss_G_L1 + loss_G_perc + loss_G_TV + loss_G_ssim
            loss_G.backward()
            optimizerG.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            steps += 1

            if i % 50 == 0 or i == 1:
                print(
                    f"Epoch [{epoch}/{cfg.epochs}] Step [{i}/{len(train_loader)}] "
                    f"D: {loss_D.item():.4f} | G: {loss_G.item():.4f} "
                    f"(GAN {loss_G_GAN.item():.4f} + L1 {loss_G_L1.item():.4f} "
                    f"+ Perc {loss_G_perc.item():.4f} + TV {loss_G_TV.item():.6f} "
                    f"+ SSIM {loss_G_ssim.item():.4f})"
                )

        avg_g_loss = epoch_g_loss / max(steps, 1)
        avg_d_loss = epoch_d_loss / max(steps, 1)
        val_l1 = validate_kaist(model, val_loader, device)
        print(
            f"Epoch [{epoch}/{cfg.epochs}] DONE | "
            f"avg D: {avg_d_loss:.4f} | avg G: {avg_g_loss:.4f} | "
            f"val L1: {val_l1:.4f}"
        )

        # Save periodic checkpoints
        if (epoch % cfg.save_every == 0) or (epoch == cfg.epochs):
            ckpt_path = os.path.join(cfg.save_dir, f"netG_epoch_{epoch:03d}.pth")
            torch.save(model.netG.state_dict(), ckpt_path)
            print(f"Saved generator checkpoint to {ckpt_path}")

        # Save best model based on validation L1
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            torch.save(model.netG.state_dict(), best_ckpt_path)
            print(f"New best model saved to {best_ckpt_path} (val L1={best_val_l1:.4f})")

        # Step LR schedulers
        schedulerG.step()
        schedulerD.step()
        current_lr_G = optimizerG.param_groups[0]["lr"]
        print(f"Current LR (G): {current_lr_G:.6e}")

    print(f"Training finished. Best val L1: {best_val_l1:.4f}, best model: {best_ckpt_path}")


# =========================================================
# 13) main
# =========================================================

def main():
    """
    Script entry point:
      - Creates a Config
      - Runs train or test depending on cfg.mode
    """
    cfg = Config()

    print("Config mode:", cfg.mode)
    print("SAVE_DIR:", cfg.save_dir)
    print("OUTPUT_DIR:", cfg.output_dir)
    print("TEST_G_WEIGHTS:", cfg.test_G_weights)

    if cfg.mode == "train":
        train_kaist(cfg)
    elif cfg.mode == "test":
        run_test(cfg)
    else:
        raise ValueError("cfg.mode must be 'train' or 'test'")


if __name__ == "__main__":
    main()
