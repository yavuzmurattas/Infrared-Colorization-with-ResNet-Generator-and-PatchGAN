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

# For perceptual loss (VGG)
from torchvision import models

# For SSIM (if available)
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


# =========================================================
# 0) Config
# =========================================================

class Config:
    def __init__(self):
        # "train"  => train on KAIST IR→RGB pairs
        # "test"   => colorize IR images from input_dir and compute metrics
        self.mode = "test"   # "train" or "test"

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Image size (images will be resized to img_size x img_size)
        self.img_size = 256

        # Input / output channel counts
        self.input_nc = 1     # IR (grayscale)
        self.output_nc = 3    # RGB
        self.ngf = 64
        self.norm = "instance"
        self.no_antialias = False
        self.no_antialias_up = False

        # ---------- TRAIN (KAIST) ----------
        # KAIST_ROOT:
        #   KAIST_ROOT/
        #     V000/lwir, V000/visible
        #     V001/lwir, V001/visible
        #     ...
        self.kaist_root = r"kaist-dataset\versions\1\set00"

        self.batch_size = 4
        self.epochs = 50
        self.lr_G = 2e-4
        self.lr_D = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lambda_L1 = 100.0
        # Perceptual + TV loss weights
        self.lambda_perc = 10.0
        self.lambda_tv = 1e-4

        self.num_workers = 4
        self.save_dir = "./checkpoints_kaist"
        self.save_every = 5
        self.val_ratio = 0.1  # percentage of dataset used for validation

        # Learning rate decay: after this epoch, linearly decay LR to 0
        self.lr_decay_start_epoch = 25  # for 50 epochs: first 25 const, last 25 decay

        # Optionally start training with a pretrained generator
        self.init_G_weights = None  # e.g., r"./pretrained_netG.pth"

        # ---------- TEST (INFERENCE + METRICS) ----------
        # input_dir should point to an "lwir" folder; visible GT is assumed to sit next to it
        self.input_dir = r"kaist-dataset\versions\1\set01\V000\lwir"
        self.output_dir = "./results"
        self.test_G_weights = r"./checkpoints_kaist/netG_best.pth"  # use best model by default

        # Top-K best output saving
        self.topk = 50
        self.best50_dirname = "Best_50_colored_images"


# =========================================================
# 1) Basic helpers + norm / init
# =========================================================

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    elif norm_type == 'none' or norm_type is None:
        return lambda num_features: Identity()
    else:
        raise NotImplementedError(f'Normalization type [{norm_type}] not supported')


def init_weights(net, init_type='normal', init_gain=0.02):
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
    net.to(device)
    if initialize_weights:
        init_weights(net, init_type, init_gain)
    return net


def get_lr_lambda(cfg: Config):
    """Linear LR decay after lr_decay_start_epoch."""
    def lr_lambda(epoch):
        # epoch is 0-based in scheduler, but we use 1-based logic in training loop
        e = epoch + 1
        if e <= cfg.lr_decay_start_epoch:
            return 1.0
        else:
            if e >= cfg.epochs:
                return 0.0
            frac = float(e - cfg.lr_decay_start_epoch) / float(max(1, cfg.epochs - cfg.lr_decay_start_epoch))
            return max(0.0, 1.0 - frac)
    return lr_lambda


# =========================================================
# 2) Anti-aliased Downsample / Upsample
# =========================================================

def get_filter(filt_size=3):
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
    """Anti-aliased upsample: bilinear + blur."""
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
# 3) ResNet Block
# =========================================================

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, use_bias=False):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

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

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
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
# 4) U-Net style ResNet Generator (IR-colorization / CUT-style)
# =========================================================

class ResnetUNetGenerator(nn.Module):
    """
    U-Net style generator:
    - Encoder: c7s1-64 -> d128 -> d256
    - Bottleneck: 9 ResNet blocks at 256 channels
    - Decoder: up256->128 with skip from 128, up128->64 with skip from 64
    """
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=9,
                 padding_type='reflect',
                 no_antialias=False, no_antialias_up=False):
        super().__init__()
        assert n_blocks >= 0
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        # Initial conv: c7s1-64
        self.inc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        )

        # Down 1: 64 -> 128, /2
        stride_d = 1 if not no_antialias else 2
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=stride_d, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
        )
        self.down1_down = None if no_antialias else Downsample(ngf * 2)

        # Down 2: 128 -> 256, /2 again
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=stride_d, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
        )
        self.down2_down = None if no_antialias else Downsample(ngf * 4)

        # ResNet blocks (bottleneck) at 256 channels
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout, use_bias)
            )
        self.resblocks = nn.Sequential(*blocks)

        # Up 1: 256 -> 256 (upsample), then concat with 128, then conv to 128
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

        # Up 2: 128 -> 128 (upsample), then concat with 64, then conv to 64
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

        # Final output conv: c7s1-3 + tanh
        self.outc = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x, layers=None, encode_only=False):
        # We ignore layers/encode_only for simplicity; keep API compatible.
        x0 = self.inc(x)           # (B, 64, H, W)
        x1 = self.down1(x0)        # (B, 128, H/2, W/2) or H, if no_antialias first
        if self.down1_down is not None:
            x1 = self.down1_down(x1)  # (B, 128, H/2, W/2)

        x2 = self.down2(x1)        # (B, 256, H/4, W/4) or etc.
        if self.down2_down is not None:
            x2 = self.down2_down(x2)  # (B, 256, H/4, W/4)

        x3 = self.resblocks(x2)    # bottleneck: (B, 256, H/4, W/4)

        y = self.up1_up(x3)        # (B, 256, H/2, W/2)
        # skip from x1
        if y.shape[-2:] != x1.shape[-2:]:
            y = F.interpolate(y, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, x1], dim=1)   # (B, 256+128, H/2, W/2)
        y = self.up1_conv(y)            # (B, 128, H/2, W/2)

        y = self.up2_up(y)              # (B, 128, H, W)
        # skip from x0
        if y.shape[-2:] != x0.shape[-2:]:
            y = F.interpolate(y, size=x0.shape[-2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, x0], dim=1)   # (B, 128+64, H, W)
        y = self.up2_conv(y)            # (B, 64, H, W)

        out = self.outc(y)              # (B, 3, H, W)
        return out, None


# =========================================================
# 5) PatchGAN Discriminator
# =========================================================

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
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
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# =========================================================
# 6) Perceptual loss (VGG-16) + TV loss
# =========================================================

class VGGPerceptual(nn.Module):
    """
    Uses VGG-16 pretrained on ImageNet.
    Input is expected in [-1,1]; it is converted to [0,1] and normalized.
    """
    def __init__(self, device):
        super().__init__()
        # torchvision >= 0.13 style weights API
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except AttributeError:
            # older version fallback
            vgg = models.vgg16(pretrained=True)

        # up to relu3_3 (approx. features[:16])
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).to(device)
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        # x: Bx3xHxW in [-1,1]
        x = (x + 1.0) / 2.0  # to [0,1]
        x = (x - self.mean) / self.std
        return self.features(x)


def tv_loss(x):
    """
    Total variation loss, encouraging spatial smoothness.
    x: BxCxHxW
    """
    diff_i = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    diff_j = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return diff_i + diff_j


# =========================================================
# 7) IRColorizationModel wrapper
# =========================================================

class IRColorizationModel(nn.Module):
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
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.netG.load_state_dict(state, strict=False)

    def forward(self, ir_tensor):
        fake_b, _ = self.netG(ir_tensor)
        return fake_b


# =========================================================
# 8) Inference helpers
# =========================================================

def load_ir_image(path, img_size=None):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    if img_size is not None:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        if img.dtype == np.uint8:
            img /= 255.0
        else:
            img /= 65535.0
    img = np.clip(img, 0.0, 1.0)
    return img


def load_rgb_image(path, img_size=None):
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
    img = img_hw[None, None, :, :]
    img = torch.from_numpy(img).float()
    img = img * 2.0 - 1.0
    return img


def tensor_to_rgb_image(tensor_bchw):
    x = tensor_bchw[0].detach().cpu().numpy()
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    return x


def save_rgb(path, img_rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img_rgb).save(path)


def collect_images(input_dir):
    exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(input_dir, ext)))
    files = sorted(files)
    return files


# =========================================================
# 9) KAIST Dataset (Vxxx/lwir, Vxxx/visible)
# =========================================================

class KAISTPairDataset(Dataset):
    """
    Expected structure:
    root/
      V000/
        lwir/
        visible/
      V001/
        lwir/
        visible/
      ...
    All Vxxx folders are scanned and IR-RGB pairs are collected.
    'indices' is used to select a subset for train/validation splits.
    """
    def __init__(self, root, img_size=256, augment=True, indices=None):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.augment = augment

        all_ir = []
        all_rgb = []

        # Also support the case where root has direct 'lwir' & 'visible' folders
        direct_ir_dir = os.path.join(root, 'lwir')
        direct_rgb_dir = os.path.join(root, 'visible')
        if os.path.isdir(direct_ir_dir) and os.path.isdir(direct_rgb_dir):
            seq_dirs = [root]
        else:
            # Subfolders: V000, V001, ...

            seq_dirs = [
                os.path.join(root, d)
                for d in sorted(os.listdir(root))
                if os.path.isdir(os.path.join(root, d))
            ]

        for seq in seq_dirs:
            ir_dir = os.path.join(seq, 'lwir')
            rgb_dir = os.path.join(seq, 'visible')
            if not (os.path.isdir(ir_dir) and os.path.isdir(rgb_dir)):
                continue

            ir_files = []
            rgb_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                ir_files.extend(glob(os.path.join(ir_dir, ext)))
                rgb_files.extend(glob(os.path.join(rgb_dir, ext)))

            ir_files = sorted(ir_files)
            rgb_files = sorted(rgb_files)

            if len(ir_files) == 0 or len(rgb_files) == 0:
                continue
            if len(ir_files) != len(rgb_files):
                print(f"[WARN] In seq {seq}: IR({len(ir_files)}) != RGB({len(rgb_files)}), skipping this seq.")
                continue

            all_ir.extend(ir_files)
            all_rgb.extend(rgb_files)

        if len(all_ir) == 0:
            raise RuntimeError(f"No IR-RGB pairs found under {root}")

        # If indices are given, select a subset (for train/val split)
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
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read IR image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        if img.max() > 1.0:
            if img.dtype == np.uint8:
                img /= 255.0
            else:
                img /= 65535.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def _read_rgb(self, path):
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

        # Simple geometric augmentation: horizontal flip
        if self.augment and random.random() < 0.5:
            ir = np.fliplr(ir).copy()
            rgb = np.fliplr(rgb).copy()

        ir_t = torch.from_numpy(ir).unsqueeze(0)                 # 1 x H x W
        rgb_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))   # 3 x H x W

        ir_t = ir_t * 2.0 - 1.0
        rgb_t = rgb_t * 2.0 - 1.0

        return {'ir': ir_t, 'rgb': rgb_t}


# =========================================================
# 10) Test (inference) mode + metrics
# =========================================================

def compute_metrics(pred_01, gt_01):
    """
    pred_01, gt_01: HxWx3 float32 images in [0,1].
    Returns: mae, mse, psnr, ssim_val
    """
    diff = pred_01 - gt_01
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20.0 * math.log10(1.0) - 10.0 * math.log10(mse + 1e-12)

    if HAVE_SKIMAGE:
        try:
            ssim_val = float(ssim(gt_01, pred_01, data_range=1.0, channel_axis=2))
        except TypeError:
            # Fallback for older scikit-image versions
            ssim_val = float(ssim(gt_01, pred_01, data_range=1.0, multichannel=True))
    else:
        ssim_val = None

    return mae, mse, psnr, ssim_val


def save_best_k_outputs(cfg: Config, metrics_list):
    """
    Copies best-K colored images from cfg.output_dir into:
        cfg.output_dir / cfg.best50_dirname
    Ranking metric:
      - SSIM if available, else PSNR
    Also writes a ranking file.
    """
    if not metrics_list:
        print("[TOP-K] metrics_list empty, skipping best-K save.")
        return

    # Prefer SSIM if available and present
    if HAVE_SKIMAGE and any(m.get("ssim") is not None for m in metrics_list):
        metric_key = "ssim"
    else:
        metric_key = "psnr"

    valid = []
    for m in metrics_list:
        v = m.get(metric_key, None)
        if v is None:
            continue
        if isinstance(v, float) and (not np.isfinite(v)):
            continue
        valid.append(m)

    if not valid:
        print(f"[TOP-K] No valid '{metric_key}' values, skipping best-K save.")
        return

    valid.sort(key=lambda x: x[metric_key], reverse=True)
    top_k = valid[:max(1, int(cfg.topk))]

    best_dir = os.path.join(cfg.output_dir, cfg.best50_dirname)
    os.makedirs(best_dir, exist_ok=True)

    rank_path = os.path.join(best_dir, f"top_{len(top_k)}_ranking.csv")
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write(f"rank,file,mae,mse,psnr,ssim,metric_used\n")
        for r, m in enumerate(top_k, start=1):
            ssim_val = m.get("ssim", None)
            ssim_str = "" if ssim_val is None else f"{ssim_val:.6f}"
            f.write(
                f"{r},{m['file']},{m['mae']:.8f},{m['mse']:.8f},{m['psnr']:.6f},{ssim_str},{metric_key}\n"
            )

    copied = 0
    for m in top_k:
        fname = m["file"]
        src = os.path.join(cfg.output_dir, fname)
        dst = os.path.join(best_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"[TOP-K][WARN] Missing output file, cannot copy: {src}")

    print(f"[TOP-K] Saved best {copied}/{len(top_k)} outputs to: {best_dir}")
    print(f"[TOP-K] Ranking file: {rank_path}")


def run_test(cfg: Config):
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

    img_paths = collect_images(cfg.input_dir)
    print(f"Found {len(img_paths)} IR images in {cfg.input_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Assume visible GT is in sibling 'visible' folder next to 'lwir'
    ir_dir = cfg.input_dir
    if os.path.basename(ir_dir).lower() == "lwir":
        vis_dir = os.path.join(os.path.dirname(ir_dir), "visible")
    else:
        vis_dir = ir_dir.replace("lwir", "visible")

    if not os.path.isdir(vis_dir):
        print(f"WARNING: visible directory not found at: {vis_dir}")
        print("Metrics will be skipped (no ground truth RGB found).")

    # Metrics accumulation
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

    for idx, path in enumerate(img_paths, start=1):
        ir = load_ir_image(path, img_size=cfg.img_size)
        ir_tensor = ir_to_tensor(ir).to(device)

        with torch.no_grad():
            fake_rgb = model(ir_tensor)

        fake_rgb_np = tensor_to_rgb_image(fake_rgb)  # uint8, HxWx3, [0..255]
        base = os.path.basename(path)
        out_path = os.path.join(cfg.output_dir, base)
        save_rgb(out_path, fake_rgb_np)

        # Metrics: only if GT exists
        gt_path = os.path.join(vis_dir, base)
        if os.path.isdir(vis_dir) and os.path.isfile(gt_path):
            gt_rgb_01 = load_rgb_image(gt_path, img_size=cfg.img_size)   # [0,1] float
            pred_rgb_01 = fake_rgb_np.astype(np.float32) / 255.0         # [0,1] float

            mae, mse, psnr_val, ssim_val = compute_metrics(pred_rgb_01, gt_rgb_01)

            metrics_list.append({
                "file": base,
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
                best_psnr_sample = base
            if ssim_val is not None and ssim_val > best_ssim:
                best_ssim = ssim_val
                best_ssim_sample = base

        else:
            if os.path.isdir(vis_dir):
                print(f"[WARN] No GT RGB found for {base} at {gt_path}; metrics skipped for this image.")

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"[{idx}/{len(img_paths)}] {path} -> {out_path}")

    print("Test finished.")

    # Write metrics to file
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
        if best_psnr_sample is not None:
            print(f"Best PSNR  : {best_psnr:.4f} ({best_psnr_sample})")
        else:
            print("Best PSNR  : N/A")
        if HAVE_SKIMAGE and best_ssim_sample is not None:
            print(f"Best SSIM  : {best_ssim:.6f} ({best_ssim_sample})")
        else:
            print("Best SSIM  : N/A")

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

        # --------- Save Best-50 colored outputs ----------
        save_best_k_outputs(cfg, metrics_list)

    else:
        print("No metrics were computed (no matching GT RGB images found).")


# =========================================================
# 11) Validation (L1 only)
# =========================================================

def validate_kaist(model: IRColorizationModel, val_loader, device):
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
# 12) Train loop (Pix2Pix-style LSGAN + L1 + Perceptual + TV)
# =========================================================

def train_kaist(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[TRAIN] Device: {device}")
    print(f"KAIST root (V000, V001, ...): {cfg.kaist_root}")

    # Collect the full dataset first
    base_dataset = KAISTPairDataset(cfg.kaist_root, img_size=cfg.img_size,
                                    augment=False, indices=None)
    N = len(base_dataset)
    val_size = max(1, int(N * cfg.val_ratio))
    train_size = N - val_size
    print(f"Total pairs: {N}, train: {train_size}, val: {val_size}")

    # Shuffle indices and create train/val split
    idxs = list(range(N))
    random.seed(42)
    random.shuffle(idxs)
    train_indices = idxs[:train_size]
    val_indices = idxs[train_size:]

    train_dataset = KAISTPairDataset(cfg.kaist_root, img_size=cfg.img_size,
                                     augment=True, indices=train_indices)
    val_dataset = KAISTPairDataset(cfg.kaist_root, img_size=cfg.img_size,
                                   augment=False, indices=val_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True, drop_last=False)

    model = IRColorizationModel(cfg)
    if cfg.init_G_weights is not None and os.path.isfile(cfg.init_G_weights):
        print(f"Initializing generator from: {cfg.init_G_weights}")
        model.load_weights(cfg.init_G_weights)

    norm_layer = get_norm_layer(cfg.norm)
    netD = NLayerDiscriminator(
        input_nc=cfg.input_nc + cfg.output_nc,
        ndf=64,
        n_layers=3,
        norm_layer=norm_layer,
    )
    netD = init_net(netD, init_type='normal', init_gain=0.02,
                    device=device, initialize_weights=True)

    optimizerG = torch.optim.Adam(model.netG.parameters(),
                                  lr=cfg.lr_G, betas=(cfg.beta1, cfg.beta2))
    optimizerD = torch.optim.Adam(netD.parameters(),
                                  lr=cfg.lr_D, betas=(cfg.beta1, cfg.beta2))

    # LR schedulers (linear decay)
    lr_lambda = get_lr_lambda(cfg)
    schedulerG = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lr_lambda)
    schedulerD = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda)

    criterionGAN = nn.MSELoss()
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

            # --- Update Discriminator D ---
            optimizerD.zero_grad()
            with torch.no_grad():
                fake_rgb = model(ir)
            real_input = torch.cat([ir, rgb], dim=1)
            fake_input = torch.cat([ir, fake_rgb], dim=1)

            pred_real = netD(real_input)
            pred_fake = netD(fake_input)

            target_real = torch.ones_like(pred_real)
            target_fake = torch.zeros_like(pred_fake)

            loss_D_real = criterionGAN(pred_real, target_real)
            loss_D_fake = criterionGAN(pred_fake, target_fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizerD.step()

            # --- Update Generator G ---
            optimizerG.zero_grad()
            fake_rgb = model(ir)
            fake_input = torch.cat([ir, fake_rgb], dim=1)
            pred_fake = netD(fake_input)

            target_real = torch.ones_like(pred_fake)
            loss_G_GAN = criterionGAN(pred_fake, target_real)
            loss_G_L1 = criterionL1(fake_rgb, rgb) * cfg.lambda_L1

            # Perceptual loss
            feat_fake = vgg_perc(fake_rgb)
            feat_real = vgg_perc(rgb)
            loss_G_perc = F.l1_loss(feat_fake, feat_real) * cfg.lambda_perc

            # TV loss
            loss_G_TV = tv_loss(fake_rgb) * cfg.lambda_tv

            loss_G = loss_G_GAN + loss_G_L1 + loss_G_perc + loss_G_TV
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
                    f"+ Perc {loss_G_perc.item():.4f} + TV {loss_G_TV.item():.6f})"
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
    cfg = Config()
    print("Config mode:", cfg.mode)

    if cfg.mode == "train":
        train_kaist(cfg)
    elif cfg.mode == "test":
        run_test(cfg)
    else:
        raise ValueError("cfg.mode must be 'train' or 'test'")


if __name__ == "__main__":
    main()
