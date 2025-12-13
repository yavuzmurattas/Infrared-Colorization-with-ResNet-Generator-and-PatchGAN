import os
import random
from glob import glob
import math

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import functools


# =========================================================
# 0) Config
# =========================================================

class Config:
    def __init__(self):
        # "train"  => train on KAIST IR→RGB pairs
        # "test"   => colorize IR images from input_dir
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
        self.num_workers = 4
        self.save_dir = "./checkpoints_kaist"
        self.save_every = 5
        self.val_ratio = 0.1  # percentage of dataset used for validation

        # Optionally start training with a pretrained generator
        self.init_G_weights = None  # e.g., r"./pretrained_netG.pth"

        # ---------- TEST (INFERENCE) ----------
        self.input_dir = r"kaist-dataset\versions\1\set01\V000\lwir"
        self.output_dir = "./results"
        self.test_G_weights = r"./checkpoints_kaist/netG_epoch_050.pth"


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


class Upsample(nn.Module):
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
# 3) ResNet Generator (IR-colorization / CUT-style)
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


class ResnetGenerator(nn.Module):
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

        model = []
        # Initial 7x7 convolution
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # Downsampling (2x)
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2,
                              kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    nn.Conv2d(ngf * mult, ngf * mult * 2,
                              kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True),
                    Downsample(ngf * mult * 2),
                ]

        # ResNet blocks
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type, norm_layer, use_dropout, use_bias)
            ]

        # Upsampling (2x)
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=2,
                        padding=1, output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    Upsample(ngf * mult),
                    nn.Conv2d(
                        ngf * mult, int(ngf * mult / 2),
                        kernel_size=3, stride=1, padding=1, bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]

        # Final output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x, layers=None, encode_only=False):
        if layers is None or len(layers) == 0:
            out = self.model(x)
            return out, None

        feat = x
        feats = []
        for layer_id, layer in enumerate(self.model):
            feat = layer(feat)
            if layer_id in layers:
                feats.append(feat)
            if layer_id == layers[-1] and encode_only:
                return None, feats
        return feat, feats


# =========================================================
# 4) PatchGAN Discriminator
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
# 5) IRColorizationModel wrapper
# =========================================================

class IRColorizationModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        norm_layer = get_norm_layer(cfg.norm)
        self.netG = ResnetGenerator(
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
# 6) Inference helpers
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
# 7) KAIST Dataset (Vxxx/lwir, Vxxx/visible)
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

        if self.augment and random.random() < 0.5:
            ir = np.fliplr(ir).copy()
            rgb = np.fliplr(rgb).copy()

        ir_t = torch.from_numpy(ir).unsqueeze(0)                 # 1 x H x W
        rgb_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))   # 3 x H x W

        ir_t = ir_t * 2.0 - 1.0
        rgb_t = rgb_t * 2.0 - 1.0

        return {'ir': ir_t, 'rgb': rgb_t}


# =========================================================
# 8) Test (inference) mode
# =========================================================

def run_test(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[TEST] Device: {device}")

    model = IRColorizationModel(cfg)
    if cfg.test_G_weights is not None and os.path.isfile(cfg.test_G_weights):
        print(f"Loading generator weights from: {cfg.test_G_weights}")
        model.load_weights(cfg.test_G_weights)
    else:
        print("WARNING: cfg.test_G_weights is None or does not exist; "
              "generator is randomly initialized, results will be meaningless.")

    model.eval()

    img_paths = collect_images(cfg.input_dir)
    print(f"Found {len(img_paths)} images in {cfg.input_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    for idx, path in enumerate(img_paths, start=1):
        ir = load_ir_image(path, img_size=cfg.img_size)
        ir_tensor = ir_to_tensor(ir).to(device)

        with torch.no_grad():
            fake_rgb = model(ir_tensor)

        fake_rgb_np = tensor_to_rgb_image(fake_rgb)
        base = os.path.basename(path)
        out_path = os.path.join(cfg.output_dir, base)
        save_rgb(out_path, fake_rgb_np)

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"[{idx}/{len(img_paths)}] {path} -> {out_path}")

    print("Test finished.")


# =========================================================
# 9) Validation (L1 only)
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
# 10) Train loop (Pix2Pix-style LSGAN + L1)
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

    criterionGAN = nn.MSELoss()
    criterionL1 = nn.L1Loss()

    os.makedirs(cfg.save_dir, exist_ok=True)

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
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()

            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            steps += 1

            if i % 50 == 0 or i == 1:
                print(
                    f"Epoch [{epoch}/{cfg.epochs}] Step [{i}/{len(train_loader)}] "
                    f"D: {loss_D.item():.4f} | G: {loss_G.item():.4f} "
                    f"(GAN {loss_G_GAN.item():.4f} + L1 {loss_G_L1.item():.4f})"
                )

        avg_g_loss = epoch_g_loss / max(steps, 1)
        avg_d_loss = epoch_d_loss / max(steps, 1)
        val_l1 = validate_kaist(model, val_loader, device)
        print(
            f"Epoch [{epoch}/{cfg.epochs}] DONE | "
            f"avg D: {avg_d_loss:.4f} | avg G: {avg_g_loss:.4f} | "
            f"val L1: {val_l1:.4f}"
        )

        if (epoch % cfg.save_every == 0) or (epoch == cfg.epochs):
            ckpt_path = os.path.join(cfg.save_dir, f"netG_epoch_{epoch:03d}.pth")
            torch.save(model.netG.state_dict(), ckpt_path)
            print(f"Saved generator checkpoint to {ckpt_path}")


# =========================================================
# 11) main
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
