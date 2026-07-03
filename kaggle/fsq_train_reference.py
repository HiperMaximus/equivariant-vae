# ============================================================================
# HISTORICAL FSQ TRAINING REFERENCE  --  READ-ONLY, NOT ACTIVE BASELINE CODE
# ============================================================================
# Verbatim extraction of the `train.py` written by cell 1 (`%%writefile
# train.py`) of the notebook `kaggle/train_runs` -- the historical working
# dual-T4 FSQ autoencoder run that trained smoothly first time. Kept here as a
# plain .py so agents can read the fast-path recipe without parsing the
# notebook JSON. The notebook launches it with:
#
#   export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 \
#     && torchrun --standalone --nproc_per_node=2 train.py
#
# It is NOT the active SIPAIM baseline: it is an FSQ autoencoder (discrete
# quantizer, PixelShuffle upsampling, Charbonnier objective), whereas the
# active baseline is a continuous denoising VAE. Do not import, run, or edit as
# repo code. It is deliberately excluded from the quality gate (ruff
# `extend-exclude`, basedpyright `exclude = ["kaggle"]`) and kept faithful to
# the original rather than reformatted.
#
# Fast-path recipe highlights (why it trained smoothly + fast on 2x T4 NCCL):
#   - torch._dynamo.config.compiled_autograd = True  (trace backward + DDP
#     all-reduce into the graph, overlap comm with backward compute)
#   - torch._dynamo.config.optimize_ddp = False       (disable DDPOptimizer
#     graph-splitting; compiled autograd handles comm instead)
#   - DDP(static_graph=True, gradient_as_bucket_view=True,
#     find_unused_parameters=False) wrapping the EAGER model FIRST
#   - torch.compile(step_fn, dynamic=False)  (compile the step fn, fullgraph=False)
#   - static shapes: drop_last=True, strict samples_per_rank % batch_size == 0
#     guard, branchless torch.where corruption, dual-branch (prob=0.30 and 0.0)
#     warmup tracing before real training
#   - AMP fp16 autocast(cache_enabled=False) with FP32 objective islands
#   - channels_last everywhere, cudnn.benchmark=True, O(1) mmap dataset
# Provenance: kaggle/train_runs (notebook), cell 1. Do not edit by hand; if the
# notebook changes, re-extract.
# ============================================================================

"""
Distributed training script for an FSQ-Autoencoder on H&E Histopathology WSIs.
Target Architecture: Dual NVIDIA T4 GPUs via DistributedDataParallel (DDP).

Objective:
Train a continuous-to-discrete FSQ Autoencoder robust to multi-center H&E staining variations. 
The discrete 16-bin bottleneck must prioritize spatial tissue morphology (we test this 
indirectly via equivariance in embedding space) while normalizing scanner and 
chemical optical density shifts (invariance to injected noise in input).

Execution Protocol: 
Execute the following command in a separate Jupyter cell after writing this file:
!export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && torchrun --standalone --nproc_per_node=2 train.py
"""

import os
import math
import json
import mmap
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import time

# Suppress benign PyTorch warnings triggered by passing read-only mmap arrays to torch.from_numpy
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# ==========================================
# 1. Configuration State & Traceability
# ==========================================
# Centralized immutable configuration dictionary. 
# Rank 0 serializes this to disk before allocating VRAM to guarantee exact 
# reproduction parameters are saved even if the script crashes during execution.
CONFIG = {
    "run_name": "run_005",  # MUST MANUALLY INCREMENT across isolated Kaggle sessions
    "debug_mode": True,    # Truncates dataset iterators to validate the full pipeline locally
    
    # Copy and paste the previous' run full path
    "resume_checkpoint": "/kaggle/input/datasets/maximusshtefan/non-eq-vae-output/run_004/checkpoints/checkpoint_ep_28.5.pt", 
    
    # Data & Hardware Constraints
    "batch_size": 60,      # Perfectly divides 150,000 patches per GPU (1200 steps exactly)
    "num_workers": 1,       # 1 worker per rank = 4 active processes total. Maximizes the 4 physical CPU cores without thrashing.
    "epochs": 35,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    
    # Domain Randomization Constraints
    "corrupt_prob": 0.30,           # 30% of batch corrupted. Forces FSQ bottleneck to prioritize clean tissue morphology.
    "corrupt_alpha": [0.75, 1.25],  # Simulates 3µm to 5µm slide slicing thickness variation.
    "corrupt_beta": [-0.10, 0.10],  # Simulates +/- 10% baseline scanner illumination variance.
    "corrupt_noise": 0.03,          # Enforces Sobolev penalty. Standard deviation matches CMOS scanner read noise.
    
    # Bottleneck Geometry
    "fsq_levels": 16,
    "latent_dim": 16,
    
    # I/O Paths
    "train_bin": "/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_train_shuffled.bin",
    "train_csv": "/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_train_shuffled.csv",
    "val_bin": "/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_ocean_valid.bin",
    "val_csv": "/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_ocean_valid.csv",
    "output_dir": "/kaggle/working"
}



def set_seed(seed: int = 42) -> None:
    """Enforces identical model initialization across distributed ranks."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ==========================================
# 2. Distributed Data Infrastructure
# ==========================================
class DistributedUBCDataset(Dataset):
    """
    O(1) disk-seeking dataset loader utilizing lazy memory mapping and OS-level madvise.
    Bypasses the Python Global Interpreter Lock (GIL) and sequential 
    f.read() overhead by mapping the binary file directly to 
    the operating system's virtual memory page cache.
    """
    def __init__(self, bin_path: str, csv_path: str, rank: int, world_size: int, batch_size: int):
        self.bin_path = bin_path
        
        # We read the CSV only to dynamically extract the total patch count
        meta_df = pd.read_csv(csv_path, usecols=["wsi_id"])
        total_length = len(meta_df)
        del meta_df 
        
        # Rank division: GPU 0 takes the first half, GPU 1 takes the second half
        samples_per_rank = total_length // world_size
        
        # Strict divisibility assertion to prevent dynamic graph recompilation
        if samples_per_rank % batch_size != 0:
            raise ValueError(
                f"Rank distribution error: samples_per_rank ({samples_per_rank}) "
                f"is not strictly divisible by batch_size ({batch_size}). "
                f"Remainder: {samples_per_rank % batch_size}. This will cause torch.compile recompilation."
            )
            
        self.start_idx = rank * samples_per_rank
        self.length = samples_per_rank
        
        self.HEADER_SIZE = 64
        
        # Deferred initialization states to prevent POSIX fork file descriptor sharing
        self.patches = None
        self.file_obj = None
        self.mm = None

    def _init_mmap(self) -> None:
        """Executes strictly inside the isolated worker process to acquire an exclusive OS file descriptor."""
        self.file_obj = open(self.bin_path, "rb")
        self.mm = mmap.mmap(self.file_obj.fileno(), length=0, access=mmap.ACCESS_READ)
        
        # Explicit OS-level command to optimize page cache for sequential reads
        self.mm.madvise(mmap.MADV_SEQUENTIAL)
        
        raw_data = np.frombuffer(self.mm, dtype=np.uint8, offset=self.HEADER_SIZE)
        
        # copy=False strictly enforces view creation; raises ValueError if contiguous memory mapping fails
        self.patches = raw_data.reshape(-1, 3, 256, 256)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.patches is None:
            self._init_mmap()
            
        global_idx = self.start_idx + idx
        # torch.from_numpy strictly adopts the underlying memory-mapped C-pointer
        return torch.from_numpy(self.patches[global_idx])

    def __del__(self):
        # Graceful file descriptor cleanup
        if hasattr(self, 'mm') and self.mm is not None:
            self.mm.close()
        if hasattr(self, 'file_obj') and self.file_obj is not None:
            self.file_obj.close()

# ==========================================
# 3. Neural Architecture & FSQ
# ==========================================
def icnr_init(tensor: torch.Tensor, upscale_factor: int = 2) -> None:
    """Initializes transposed convolutions to mitigate checkerboard frequency artifacts."""
    out_channels, in_channels, kernel_h, kernel_w = tensor.shape
    sub_out_channels = out_channels // (upscale_factor ** 2)
    sub_tensor = torch.empty(sub_out_channels, in_channels, kernel_h, kernel_w)
    init.kaiming_normal_(sub_tensor, mode="fan_out", nonlinearity="relu")
    sub_tensor = sub_tensor.repeat_interleave(upscale_factor ** 2, dim=0)
    with torch.no_grad():
        tensor.copy_(sub_tensor)

class Stem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn = nn.GroupNorm(32, 64)
        self.relu = nn.ReLU(inplace=True)
        init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        init.constant_(self.gn.weight, 1)
        init.constant_(self.gn.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gn(self.conv(x)))

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, out_channels)
        self.spatial_downsample = None
        
        if stride != 1 or in_channels != out_channels:
            if stride != 1:
                # ResNet-D specification: Anti-aliasing spatial aggregation prior to channel projection
                self.spatial_downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
            else:
                self.spatial_downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.GroupNorm(32, out_channels),
                )
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d): init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm): 
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.spatial_downsample(x) if self.spatial_downsample is not None else x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)

class Encoder(nn.Module):
    """Refactored isolated encoder terminating directly before quantization bounds."""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.stem = Stem()
        self.enc_layer1 = self._make_enc_layer(64, 64, blocks=2, stride=1)
        self.enc_layer2 = self._make_enc_layer(64, 128, blocks=2, stride=2)
        self.enc_layer3 = self._make_enc_layer(128, 256, blocks=2, stride=2)
        self.enc_layer4 = self._make_enc_layer(256, 512, blocks=2, stride=1)
        self.enc_norm = nn.GroupNorm(32, 512)
        self.enc_act = nn.SiLU(inplace=True)
        self.enc_out = nn.Conv2d(512, latent_dim, kernel_size=3, padding=1)

    @staticmethod
    def _make_enc_layer(in_c: int, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [EncoderBlock(in_c, out_c, stride)]
        layers.extend([EncoderBlock(out_c, out_c, 1) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_layer4(self.enc_layer3(self.enc_layer2(self.enc_layer1(self.stem(x)))))
        # Cast strictly prior to the final spatial projection to avert AMP FP16 truncation
        with torch.amp.autocast('cuda', enabled=False):
            return self.enc_out(self.enc_act(self.enc_norm(x)).to(torch.float32))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 1) -> None:
        super().__init__()
        
        if upscale_factor > 1:
            expanded_channels = out_channels * (upscale_factor ** 2)
            # GroupNorm applied before spatial rearrangement to prevent zero-variance collapse on ICNR sub-pixels
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, expanded_channels),
                nn.PixelShuffle(upscale_factor),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, expanded_channels),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels)
            ) if in_channels != out_channels else None

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d): init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm): 
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
        if upscale_factor > 1:
            icnr_init(self.conv1[0].weight, upscale_factor=upscale_factor)
            if self.shortcut is not None: icnr_init(self.shortcut[0].weight, upscale_factor=upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv2(self.conv1(x))
        return self.relu(out + identity)

class Decoder(nn.Module):
    """Refactored isolated decoder expecting bounded normalized continuous latents."""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.dec_in = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1, bias=False)
        self.dec_layer4 = self._make_dec_layer(512, 256, blocks=2, upscale_factor=1)
        self.dec_layer3 = self._make_dec_layer(256, 128, blocks=2, upscale_factor=2)
        self.dec_layer2 = self._make_dec_layer(128, 64, blocks=2, upscale_factor=2)
        self.dec_layer1 = self._make_dec_layer(64, 64, blocks=2, upscale_factor=2)
        self.dec_norm = nn.GroupNorm(32, 64)
        self.dec_act = nn.SiLU(inplace=True)
        self.dec_out = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True)
        
        init.kaiming_normal_(self.dec_in.weight, mode="fan_out", nonlinearity="relu")
        init.constant_(self.dec_norm.weight, 1)
        init.constant_(self.dec_norm.bias, 0)
        
        # Ensures initial neutral predictions before gradient accumulation establishes structure
        init.constant_(self.dec_out.weight, 0)
        init.constant_(self.dec_out.bias, 0)

    @staticmethod
    def _make_dec_layer(in_c: int, out_c: int, blocks: int, upscale_factor: int) -> nn.Sequential:
        layers = [DecoderBlock(in_c, out_c, upscale_factor)]
        layers.extend([DecoderBlock(out_c, out_c, 1) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec_layer1(self.dec_layer2(self.dec_layer3(self.dec_layer4(self.dec_in(z)))))
        logits = self.dec_out(self.dec_act(self.dec_norm(x)))
        # Strictly returns FP32 bounded logits to maintain Charbonnier objective precision
        return torch.tanh(logits.to(torch.float32))

class FiniteScalarQuantizer(nn.Module):
    """
    Implements continuous-to-discrete bounding. Projects a continuous 
    latent variable onto L uniformly spaced integer bins.
    """
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.L = CONFIG["fsq_levels"]
        
        # Initialized exactly to 0.0, ensuring the mathematical scale evaluates to s = 1.0.
        # Capping with sigmoid prevents s from blowing up and killing the tanh gradients.
        self.s_raw = nn.Parameter(torch.zeros(latent_dim))

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z.to(torch.float32)
        s = 2.0 * torch.sigmoid(self.s_raw.view(1, -1, 1, 1))
        
        z_scaled = s * torch.tanh(z)
        
        # Straight-Through Estimator (STE) bypasses domain clamping gradient starvation
        z_bounded = torch.clamp(z_scaled, min=-1.0, max=1.0)
        z_bounded = z_scaled + (z_bounded - z_scaled).detach()
        
        z_projected = ((z_bounded + 1.0) / 2.0) * (self.L - 1)
        z_rounded = torch.round(z_projected)
        
        # Straight-Through Estimator (STE) bypasses the non-differentiable round()
        z_q = z_projected + (z_rounded - z_projected).detach()
        z_normalized = (z_q / ((self.L - 1) / 2.0)) - 1.0
        
        return z_normalized, z_rounded.to(torch.int64), z_scaled

class FSQAutoencoder(nn.Module):
    """
    Orchestrates the refactored Encoder, Quantizer, and Decoder modules.
    Provides explicit isolated methods (`encode`, `decode`) to bypass torch.compile 
    limitations during spatial equivariance evaluation loops.
    """
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.fsq = FiniteScalarQuantizer(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Isolated structural pass. Invisible to torch.compile static graphs."""
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Isolated structural pass. Invisible to torch.compile static graphs."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_cont = self.encode(x)
        z_normalized, z_indices, z_scaled = self.fsq(z_cont)
        recon = self.decode(z_normalized)
        return recon, z_indices, z_scaled

# ==========================================
# 4. Domain Randomization Module
# ==========================================
class HistopathologyCorruptor(nn.Module):
    """
    Simulates physical H&E staining variations strictly in Beer-Lambert optical density space.
    """
    def __init__(self) -> None:
        super().__init__()
        rgb_from_hed = torch.tensor([
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776]
        ], dtype=torch.float32)
        hed_from_rgb = torch.linalg.inv(rgb_from_hed)
        self.register_buffer("hed_from_rgb", hed_from_rgb)
        self.register_buffer("rgb_from_hed", rgb_from_hed)

        # Alpha (multiplicative scalar on Optical Density). Range corresponds physically to 3µm-5µm cuts.
        a_min, a_max = CONFIG["corrupt_alpha"]
        self.register_buffer("alpha_min", torch.tensor([a_min, a_min, 0.98]).view(1, 3, 1, 1))
        self.register_buffer("alpha_max", torch.tensor([a_max, a_max, 1.02]).view(1, 3, 1, 1))
        
        # Beta (additive shift on Optical Density). Translates to uniform background scanner illumination shift.
        b_min, b_max = CONFIG["corrupt_beta"]
        self.register_buffer("beta_min", torch.tensor([b_min, b_min, -0.01]).view(1, 3, 1, 1))
        self.register_buffer("beta_max", torch.tensor([b_max, b_max,  0.01]).view(1, 3, 1, 1))

    @staticmethod
    def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.04045, x / 12.92, torch.pow(torch.clamp((x + 0.055) / 1.055, min=1e-8), 2.4))

    @staticmethod
    def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(torch.clamp(x, min=1e-8), 1.0 / 2.4) - 0.055)

    def forward(self, x: torch.Tensor, noise_std: float = 0.03) -> torch.Tensor:
        # Prevents exp/log underflows from destroying the gradient graph
        with torch.no_grad():
            dtype = x.dtype
            x = x.to(torch.float32)
            
            x_linear = self._srgb_to_linear((x + 1.0) / 2.0)
            od = -torch.log(torch.clamp(x_linear, min=1e-8))
            
            # MODIFIED: Removed premature torch.clamp(od, min=0.0, max=3.0) 
            # Rationale: Bounding prematurely distorts density geometry before affine variation.
            
            hed = torch.einsum('ij, b j h w -> b i h w', self.hed_from_rgb, od)
            
            rand_alpha = torch.rand(x.shape[0], 3, 1, 1, device=x.device, dtype=torch.float32)
            alpha = self.alpha_min + rand_alpha * (self.alpha_max - self.alpha_min)
            rand_beta = torch.rand(x.shape[0], 3, 1, 1, device=x.device, dtype=torch.float32)
            beta = self.beta_min + rand_beta * (self.beta_max - self.beta_min)
            
            hed_aug = (hed * alpha) + beta
            od_aug = torch.einsum('ij, b j h w -> b i h w', self.rgb_from_hed, hed_aug)
            
            # MODIFIED: Physical transmission bounds enforced strictly post-augmentation
            od_aug = torch.clamp(od_aug, min=0.0)
            
            x_aug_linear = torch.exp(-od_aug)
            
            x_aug = (self._linear_to_srgb(x_aug_linear) * 2.0) - 1.0
            
            # Unconditional Sobolev regularization injected strictly into the sRGB spatial domain.
            # Branchless execution preserves static compiler graphs.
            x_aug = x_aug + torch.randn_like(x_aug) * noise_std
            
            return torch.clamp(x_aug, min=-1.0, max=1.0).to(dtype, memory_format=torch.channels_last)

# ==========================================
# 5. Objectives & Metrics
# ==========================================
class CharbonnierLoss(nn.Module):
    """Robust differentiable L1 approximation objective."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps ** 2
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))

class StatelessSSIM(nn.Module):
    """Stateless functional SSIM with internal domain projection compatible with TorchDynamo JIT tracing."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, val_range: tuple = (-1.0, 1.0), channels: int = 3):
        super().__init__()
        # C1 and C2 strictly assume a data range of 1.0 post-projection
        self.C1 = (0.01 * 1.0) ** 2
        self.C2 = (0.03 * 1.0) ** 2
        
        self.val_min = val_range[0]
        self.val_max = val_range[1]
        self.val_span = self.val_max - self.val_min
        
        coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        kernel_2d = (g.unsqueeze(0) * g.unsqueeze(1)).expand(channels, 1, window_size, window_size).contiguous()
        self.register_buffer("kernel", kernel_2d)
        self.groups = channels
        self.pad = window_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Internal normalization into [0.0, 1.0] physical luminance domain
        pred_proj = (pred - self.val_min) / self.val_span
        target_proj = (target - self.val_min) / self.val_span

        pred_pad = F.pad(pred_proj, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        target_pad = F.pad(target_proj, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        
        mu1 = F.conv2d(pred_pad, self.kernel, groups=self.groups)
        mu2 = F.conv2d(target_pad, self.kernel, groups=self.groups)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        
        sigma1_sq = F.conv2d(pred_pad * pred_pad, self.kernel, groups=self.groups) - mu1_sq
        sigma2_sq = F.conv2d(target_pad * target_pad, self.kernel, groups=self.groups) - mu2_sq
        sigma12 = F.conv2d(pred_pad * target_pad, self.kernel, groups=self.groups) - mu1_mu2
        
        num = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        return (num / den).mean()

class CombinedReconstructionLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_metric = StatelessSSIM(val_range=(-1.0, 1.0), channels=3)
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_c = self.charbonnier(pred, target)
        loss_s = 1.0 - self.ssim_metric(pred, target)
        return loss_c + (self.ssim_weight * loss_s), loss_c, loss_s

# ==========================================
# 6. State Management & Logging Math
# ==========================================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_loss, filepath):
    state = {
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
        'np_rng_state': np.random.get_state()
    }
    torch.save(state, filepath, pickle_protocol=5)

# ADDED: State restoration function strictly mapping tensors to the local GPU to prevent VRAM overflow.
def load_checkpoint(filepath: str, model, optimizer, scheduler, scaler, local_rank: int):
    """Restores distributed training state from a serialized checkpoint."""
    device_map = {'cuda:0': f'cuda:{local_rank}'}
    state = torch.load(filepath, map_location=device_map,weights_only=False)
    
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    
    torch.set_rng_state(state['rng_state'])
    torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    np.random.set_state(state['np_rng_state'])
    
    return state['epoch'], state['step'], state['best_loss']

def get_parameter_groups(model, weight_decay: float = 1e-5):
    """Enforces exclusive weight decay application strictly to multi-dimensional matrices."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

def append_to_csv(data_list, filepath):
    if not data_list: return
    df = pd.DataFrame(data_list)
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, mode='w', header=True, index=False)
    data_list.clear()

class VarianceAccumulator:
    """In-memory distributed metrics accumulator avoiding implicit host-blocking .item() synchronization."""
    def __init__(self, device):
        self.device = device
        self.metrics = ['loss', 'charbonnier', 'ssim', 'mse', 'mae']
        self.reset()
        
    def reset(self):
        self.data = torch.zeros(len(self.metrics) * 2 + 1, dtype=torch.float64, device=self.device)
        
    def update(self, vals: dict):
        idx = 0
        with torch.no_grad():
            for m in self.metrics:
                # Explicit float64 cast strictly prior to addition prevents precision cancellation
                val = vals[m].to(torch.float64) 
                self.data[idx] += val
                self.data[idx + 1] += val ** 2
                idx += 2
            self.data[-1] += 1.0
        
    def reduce_and_compute(self):
        dist.all_reduce(self.data, op=dist.ReduceOp.SUM)
        global_count = self.data[-1].item()
        
        if global_count == 0:
            return {f"{m}_{stat}": 0.0 for m in self.metrics for stat in ['mu', 'sigma']}
            
        results = {}
        idx = 0
        for m in self.metrics:
            g_sum = self.data[idx].item()
            g_sq_sum = self.data[idx+1].item()
            idx += 2
            
            mu = g_sum / global_count
            var = max(0.0, (g_sq_sum / global_count) - (mu ** 2))
            
            results[f"{m}_mu"] = mu
            results[f"{m}_sigma"] = math.sqrt(var)
            
        results['psnr_mu'] = 10.0 * math.log10((2.0 ** 2) / results['mse_mu']) if results['mse_mu'] > 0 else 0.0
        return results

# ==========================================
# 7. Main Training Loop
# ==========================================
def main():
    print(f"[PROCESS START] Interpreter active at {time.strftime('%X')}. Initializing DDP...", flush=True)
    # 1. Initialize Distributed Data Parallel (DDP) Backend
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", device_id=local_rank)

    set_seed(42)

    # Enable PyTorch 2.9 Compiled Autograd to natively overlap DDP communication
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.optimize_ddp = False

    # 2. Establish File System Hierarchy & Serialize Configuration
    RUN_DIR = f"{CONFIG['output_dir']}/{CONFIG['run_name']}"
    if local_rank == 0:
        os.makedirs(RUN_DIR, exist_ok=True)
        # Writes immediately to guarantee parameters are saved even if dataset crashes
        with open(f"{RUN_DIR}/hyperparameter_config.json", "w") as f:
            json.dump(CONFIG, f, indent=4)

    BATCH_SIZE = CONFIG["batch_size"]
    
    # 3. Instantiate O(1) Memory-Mapped Datasets
    t0 = time.time()
    train_ds = DistributedUBCDataset(CONFIG["train_bin"], CONFIG["train_csv"], local_rank, world_size, BATCH_SIZE)
    val_ds = DistributedUBCDataset(CONFIG["val_bin"], CONFIG["val_csv"], local_rank, world_size, BATCH_SIZE)
    t1 = time.time()
    if local_rank == 0:
        print(f"[RANK 0] Dataset memory mapping complete in {t1 - t0:.2f} seconds. Instantiating model...", flush=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)

    model_eager = FSQAutoencoder(latent_dim=CONFIG["latent_dim"]).to(device)
    # Physically reorder weight bytes to optimal NHWC memory layout for Tensor Cores
    model_eager = model_eager.to(memory_format=torch.channels_last)
    
    corruptor = HistopathologyCorruptor().to(device).to(memory_format=torch.channels_last)
    criterion = CombinedReconstructionLoss().to(device).to(memory_format=torch.channels_last)
    
    optim_groups = get_parameter_groups(model_eager, weight_decay=CONFIG["weight_decay"])
    optimizer = torch.optim.AdamW(optim_groups, lr=CONFIG["lr"])
    scaler = torch.amp.GradScaler('cuda')
    
    TOTAL_EPOCHS = CONFIG["epochs"]
    total_steps = len(train_loader) * TOTAL_EPOCHS
    warmup_steps = int(total_steps * 0.05)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(train_loader) * 10, T_mult=1, eta_min=1e-6
    )

    # ADDED: Checkpoint Resumption Protocol prior to DDP compilation
    resume_ckpt = CONFIG.get("resume_checkpoint", "")
    if resume_ckpt and os.path.exists(resume_ckpt):
        if local_rank == 0: print(f"[RANK 0] Restoring state from {resume_ckpt}", flush=True)
            
        start_epoch, step, best_val_loss = load_checkpoint(resume_ckpt, model_eager, optimizer, scheduler, scaler, local_rank)
        dist.barrier()
        
        is_midpoint = ".5" in os.path.basename(resume_ckpt)
        resume_batch_idx = (len(train_loader) // 2) if is_midpoint else 0
    else:
        start_epoch, step, best_val_loss, resume_batch_idx = 0, 0, float('inf'), 0

    # 4. Neural Graph Compilation & Orchestrator Setup
    # DDP wraps the eager module first to allow compiled autograd to trace communication hooks
    model = DDP(model_eager, device_ids=[local_rank], 
                static_graph=True, 
                gradient_as_bucket_view=True,
                find_unused_parameters=False,)
    
    noise_std = float(CONFIG["corrupt_noise"])
    corrupt_prob = float(CONFIG["corrupt_prob"])
    
    # MODIFIED: Branchless orchestrator executing corruption over the entire batch 
    # to strictly satisfy dynamic=False while randomizing corrupted indices via a boolean mask.
    def step_fn(x_clean: torch.Tensor, c_prob: float):
        x_corrupt = corruptor(x_clean, noise_std=noise_std)
        mask = torch.rand(x_clean.shape[0], 1, 1, 1, device=x_clean.device, dtype=x_clean.dtype) < c_prob
        x_in = torch.where(mask, x_corrupt, x_clean)
        
        # cache_enabled=False strictly required for CUDA graphs within AMP context
        with torch.amp.autocast('cuda', dtype=torch.float16, cache_enabled=False):
            recon_images, z_indices, z_scaled = model(x_in)
            
        # FP32 Objective executes strictly isolated from the AMP context
        recon_fp32 = recon_images.to(torch.float32)
        total_recon_loss, loss_c, loss_s = criterion(recon_fp32, x_clean)
        # Linear (L1) Gravitational Penalty strictly isolates momentum shock
        gamma = 0.1
        excursion_loss = gamma * torch.mean(F.relu(torch.abs(z_scaled) - 1.0))
        
        # Optimization target
        optimization_loss = total_recon_loss + excursion_loss
        
        # Return optimization_loss for backprop, and total_recon_loss for isolated telemetry
        return optimization_loss, total_recon_loss, loss_c, loss_s, recon_fp32, z_indices

    # fullgraph=False (default) permits TorchDynamo to manage control flow boundaries and communication overlapping
    compiled_step = torch.compile(step_fn, dynamic=False)

    # 5. Graph Compilation Warmup (Dual Branch Tracing)
    model.train()
    dist.barrier()
    
    print(f"[RANK {local_rank}] Got to warmup.", flush=True)
    # MODIFIED: Warmup tracing iterates over the exact scalar probabilities the orchestrator will encounter
    for prob in [corrupt_prob, 0.0]:
        for _ in range(3):
            # Formulate uniform noise bounded in [-1.0, 1.0] across the spatial field to trace all AMP activation ranges
            dummy_spatial = torch.linspace(-1.0, 1.0, steps=256 * 256, device=device).view(256, 256)
            dummy_clean = dummy_spatial.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, 3, 256, 256).contiguous()
            dummy_clean = dummy_clean.to(memory_format=torch.channels_last)
            
            dummy_total_loss, _, _, _, _, _ = compiled_step(dummy_clean, prob)
            dummy_total_loss.backward()
            
            for param in model.parameters():
                if param.grad is not None: param.grad = None
            
    del dummy_spatial, dummy_clean, dummy_total_loss
    print(f"[RANK {local_rank}] Finished warmup.", flush=True)
    torch.cuda.empty_cache()
    
    checkpoint_history = []
    telemetry_data = torch.zeros(4, dtype=torch.float64, device=device)
    train_bin_counts = torch.zeros((16, 16), dtype=torch.int64, device=device)
    val_bin_counts = torch.zeros((16, 16), dtype=torch.int64, device=device)
    # Pre-allocate and pre-reshape the offset tensor once per physical device
    # Explicitly cast to int64 to match the expected dtype of z_indices
    bin_offsets = (torch.arange(16, device=device, dtype=torch.int64) * 16).view(1, 16, 1, 1)
    
    train_acc = VarianceAccumulator(device)
    val_acc = VarianceAccumulator(device)
    epoch_stats_df, dynamics_df = [], []

    # 6. Archive Original 25-Patch References
    # These represent the immutable structural baseline for all downstream equivariance evaluations
    fixed_val_batch = None
    n_images = min(BATCH_SIZE,25)
    if local_rank == 0:
        fixed_val_batch = next(iter(val_loader)).to(device, memory_format=torch.channels_last, non_blocking=True).to(torch.float32) / 127.5 - 1.0
        ref_dir = f"{RUN_DIR}/original_reference_patches"
        os.makedirs(ref_dir, exist_ok=True)
        grid_baseline_og = (fixed_val_batch[:n_images] + 1.0) / 2.0
        for i in range(n_images):
            vutils.save_image(grid_baseline_og[i], f"{ref_dir}/original_img_{i:02d}.png")

    # ------------------------------------------
    # Execution Architecture Loop
    # ------------------------------------------
    if CONFIG["debug_mode"]:
        TOTAL_EPOCHS = start_epoch + 1
    dist.barrier()
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[RANK 0] Commencing execution from Epoch {start_epoch}, absolute batch index {resume_batch_idx}.", flush=True)
        print(f"[RANK 0] Restored Global Step: {step} | Initial LR: {current_lr:.3e}", flush=True)
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        
        # ADDED: O(1) Dataloader Fast-Forwarding via torch.utils.data.Subset
        if epoch == start_epoch and resume_batch_idx > 0:
            resume_sample_idx = resume_batch_idx * BATCH_SIZE
            indices = list(range(resume_sample_idx, len(train_ds)))
            subset_ds = torch.utils.data.Subset(train_ds, indices)
            current_loader = DataLoader(subset_ds, batch_size=BATCH_SIZE, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
        else:
            current_loader = train_loader
            
        for relative_batch_idx, images_uint8 in enumerate(current_loader):
            # ADDED: Reconstruct the absolute batch index to correctly evaluate execution boundaries
            actual_batch_idx = relative_batch_idx + (resume_batch_idx if epoch == start_epoch else 0)
            
            # Dynamic PCIe stride mapping: Logical CHW -> Physical NHWC
            x_clean = images_uint8.to(device, memory_format=torch.channels_last, non_blocking=True).to(torch.float32) / 127.5 - 1.0
            
            optimizer.zero_grad(set_to_none=True)
            
            # Unpack the updated step_fn signature
            total_loss, recon_loss, loss_c, loss_s, recon_fp32, z_indices = compiled_step(x_clean, corrupt_prob)
            
            scaler.scale(total_loss).backward()
            if CONFIG["debug_mode"] and relative_batch_idx == 0:
                for name, p in model.named_parameters():
                    if p.grad is None:
                        print(f"[RANK {local_rank}] {name} grad is None", flush=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=True)
            
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()
            new_scale = scaler.get_scale()
            
            # Identifies iterations where GradScaler aborted the step due to FP16 Inf/NaN overflow
            step_successful = new_scale >= old_scale
            
            # MODIFIED: Scheduler step is strictly dependent on step_successful, enforcing phase synchronization
            if step < warmup_steps:
                lr_scale = float(step) / float(max(1, warmup_steps))
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * pg['initial_lr']
            elif step_successful:
                scheduler.step()
                
            # Strictly gate metric accumulation to prevent NaN poisoning from aborted steps
            if step_successful:
                with torch.no_grad():
                    train_acc.update({
                        'loss': recon_loss.detach(), 'charbonnier': loss_c.detach(), 
                        'ssim': (1.0 - loss_s).detach(), 'mse': F.mse_loss(recon_fp32, x_clean), 'mae': F.l1_loss(recon_fp32, x_clean)
                    })
                    
                step += 1
                
                # 7. Telemetry & FSQ Codebook Tracking
                with torch.no_grad():
                    # ---------------------------------------------------------
                    # VECTORIZED CODEBOOK TRACKING
                    # ---------------------------------------------------------
                    # C = 16 channels, L = 16 levels. Total continuous domain = 256 bins.
                    
                    # Step A: Construct the 1D offset vector [0, 16, 32, ..., 240]
                    # which was the construction of bin_offsets previously
                    # Data type must match z_indices to avert implicit casting overhead.
                    
                    # Step B: Domain Translation via Broadcasting
                    # bin_offsets aligns with z_indices shape [B, 16, H, W].
                    # The addition shifts Channel 0 to [0, 15], Channel 1 to [16, 31], etc.
                    z_shifted = z_indices + bin_offsets
                    
                    # Step C: Fused Kernel Execution
                    # Flattening merges all shifted spatial domains into a 1D tensor.
                    # A single atomic bincount kernel computes all 256 (16*16) frequencies simultaneously.
                    fused_counts = torch.bincount(z_shifted.flatten(), minlength=256)
                    
                    # Step D: Matrix Reconstitution
                    # Reshaping the 256-element vector back to [16, 16] guarantees row `c`
                    # strictly contains the isolated frequency distribution for channel `c`.
                    train_bin_counts += fused_counts.view(16, 16)
                    # ---------------------------------------------------------

                    # Extract parameter from uncompiled reference to bypass TorchDynamo JIT attribute masking
                    s_vals = 2.0 * torch.sigmoid(model_eager.fsq.s_raw).detach()
                    telemetry_data[0] += total_loss.detach().to(torch.float64)
                    telemetry_data[1] += total_loss.detach().to(torch.float64) ** 2
                    telemetry_data[2] += s_vals.mean().to(torch.float64)
                    telemetry_data[3] += (s_vals ** 2).mean().to(torch.float64)

            # Execution boundaries 
            is_midpoint = (actual_batch_idx == (len(train_loader) // 2) - 1)
            is_endpoint = (actual_batch_idx == len(train_loader) - 1)
            if CONFIG["debug_mode"] and relative_batch_idx == 3: is_endpoint = True
            trigger_telemetry = (step % 100 == 0) or (CONFIG["debug_mode"] and is_endpoint)

            if trigger_telemetry and step_successful:
                dist.all_reduce(telemetry_data, op=dist.ReduceOp.SUM)
                
                # Dynamic denominator ensures math holds even if debug truncates the step count
                steps_accum = 100.0 if step % 100 == 0 else float(step % 100)
                N = steps_accum * world_size
                
                g_loss_sum, g_loss_sq = telemetry_data[0].item(), telemetry_data[1].item()
                g_s_sum, g_s_sq = telemetry_data[2].item(), telemetry_data[3].item()
                
                global_loss_mean = g_loss_sum / N
                global_s_mean = g_s_sum / N
                
                if local_rank == 0:
                    dynamics_df.append({
                        "step": step, 
                        "loss_mean": global_loss_mean, 
                        "loss_std": math.sqrt(max(0.0, (g_loss_sq / N) - (global_loss_mean ** 2))),
                        "lr": optimizer.param_groups[0]['lr'], 
                        "s_mean": global_s_mean, 
                        "s_std": math.sqrt(max(0.0, (g_s_sq / N) - (global_s_mean ** 2)))
                    })
                    append_to_csv(dynamics_df, f"{RUN_DIR}/train_dynamics.csv")
                telemetry_data.zero_()
                
            # ==========================================
            # 8. Evaluation Boundary (0.5 Epochs)
            # ==========================================
            if is_midpoint or is_endpoint:
                fraction_print = "5" if is_midpoint else "0"
                ep_print = str(epoch) if is_midpoint else str(epoch + 1)
                
                if local_rank == 0:
                    print(f"\n[RANK 0] --- EVALUATION START: Epoch {ep_print}.{fraction_print} ---", flush=True)
                model.eval()
                val_acc.reset()
                
                with torch.no_grad():
                    for v_batch_idx, v_batch in enumerate(val_loader):
                        if CONFIG["debug_mode"] and v_batch_idx == 2: break
                        
                        v_clean = v_batch.to(device, memory_format=torch.channels_last, non_blocking=True).to(torch.float32) / 127.5 - 1.0
                        
                        # Leveraging the compiled orchestrator function directly via the inference cache branch
                        total_loss, v_loss, v_loss_c, v_loss_s, v_recon_fp32, v_z_indices = compiled_step(v_clean, 0.0)
                            
                        val_acc.update({
                            'loss': v_loss.detach(), 'charbonnier': v_loss_c.detach(),
                            'ssim': (1.0 - v_loss_s).detach(), 'mse': F.mse_loss(v_recon_fp32, v_clean), 'mae': F.l1_loss(v_recon_fp32, v_clean)
                        })
                        # ---------------------------------------------------------
                        # VECTORIZED CODEBOOK TRACKING (VALIDATION)
                        # ---------------------------------------------------------
                        v_z_shifted = v_z_indices + bin_offsets
                        v_fused_counts = torch.bincount(v_z_shifted.flatten(), minlength=256)
                        val_bin_counts += v_fused_counts.view(16, 16)
                        # ---------------------------------------------------------
                
                # Synchronization Point 1: Global metrics reduction
                print(f"[RANK {local_rank}] Validation inference complete. Waiting for all_reduce (Global metrics reduction).", flush=True)
                t_stats = train_acc.reduce_and_compute()
                v_stats = val_acc.reduce_and_compute()
                train_acc.reset() 
                
                dist.all_reduce(train_bin_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_bin_counts, op=dist.ReduceOp.SUM)

                # ==========================================
                # 9. Spatial Equivariance Evaluation Protocol
                # ==========================================
                """
                Evaluates latent space equivariance under spatial transformations prior to quantization.
                All I/O is routed to a _tmp directory, then atomically renamed to prevent artifact 
                collision in the event of an arbitrary script death.
                
                /kaggle/working/<RUN_NAME>/
                └── evaluations/
                    └── ep_<EPOCH>.<FRACTION>/               
                        ├── continuous_latents/              
                        ├── discrete_indices/                
                        └── reconstructions/                 
                """
                if local_rank == 0:
                    print(f"[RANK 0] Metric reduction successful. Initiating Equivariance Protocol to _tmp...", flush=True)
                    final_eval_base_dir = f"{RUN_DIR}/evaluations/ep_{ep_print}.{fraction_print}"
                    tmp_eval_base_dir = f"{final_eval_base_dir}_tmp"
                    
                    cont_dir = f"{tmp_eval_base_dir}/continuous_latents"
                    disc_dir = f"{tmp_eval_base_dir}/discrete_indices"
                    recon_dir = f"{tmp_eval_base_dir}/reconstructions"
                    
                    os.makedirs(cont_dir, exist_ok=True)
                    os.makedirs(disc_dir, exist_ok=True)
                    os.makedirs(recon_dir, exist_ok=True)
                    
                    with torch.no_grad():
                        # Slicing the evaluation batch circumvents eager mode MAC overhead
                        val_subset = fixed_val_batch[:n_images]
                        
                        with torch.amp.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                            z_cont_baseline = model_eager.encode(val_subset)
                            z_norm_baseline, z_indices_baseline, _ = model_eager.fsq(z_cont_baseline)
                            recon_baseline = model_eager.decode(z_norm_baseline)
                            
                        np.save(f"{cont_dir}/latent_cont_ep_{ep_print}.{fraction_print}_baseline.npy", z_cont_baseline.cpu().numpy())
                        np.save(f"{disc_dir}/latent_indices_ep_{ep_print}.{fraction_print}_baseline.npy", z_indices_baseline.to(torch.uint8).cpu().numpy())
                        
                        grid_baseline = (recon_baseline + 1.0) / 2.0
                        for i in range(n_images):
                            vutils.save_image(grid_baseline[i], f"{recon_dir}/recon_ep_{ep_print}.{fraction_print}_img_{i:02d}_baseline.png")
                        
                        angles = {90: 1, 180: 2, 270: 3}
                        eq_errors = []
                        
                        for angle_deg, k in angles.items():
                            x_rot = torch.rot90(val_subset, k, dims=[2, 3])
                            
                            with torch.amp.autocast('cuda', dtype=torch.float16, cache_enabled=False):
                                z_cont_rot_input = model_eager.encode(x_rot)
                                z_norm_rot_input, z_indices_rot_input, _ = model_eager.fsq(z_cont_rot_input)
                                recon_from_rot_input = model_eager.decode(z_norm_rot_input)
                                
                                # Rotates the baseline continuous spatial map
                                z_cont_rot_latent = torch.rot90(z_cont_baseline, k, dims=[2, 3])
                                z_norm_rot_latent, z_indices_rot_latent, _ = model_eager.fsq(z_cont_rot_latent)
                                recon_from_rot_latent = model_eager.decode(z_norm_rot_latent)
                                    
                            # L2 Equivariance Error with numerical stability
                            num = torch.sum((z_cont_rot_latent - z_cont_rot_input) ** 2, dim=(1, 2, 3))
                            den = torch.sum(z_cont_rot_input ** 2, dim=(1, 2, 3)) + 1e-8
                            eq_errors.append((num / den).mean().item())
                            
                            # Serialization of artifacts
                            np.save(f"{cont_dir}/latent_cont_ep_{ep_print}.{fraction_print}_rot_input_{angle_deg}.npy", z_cont_rot_input.cpu().numpy())
                            np.save(f"{cont_dir}/latent_cont_ep_{ep_print}.{fraction_print}_rot_latent_{angle_deg}.npy", z_cont_rot_latent.cpu().numpy())
                            np.save(f"{disc_dir}/latent_indices_ep_{ep_print}.{fraction_print}_rot_input_{angle_deg}.npy", z_indices_rot_input.to(torch.uint8).cpu().numpy())
                            np.save(f"{disc_dir}/latent_indices_ep_{ep_print}.{fraction_print}_rot_latent_{angle_deg}.npy", z_indices_rot_latent.to(torch.uint8).cpu().numpy())
                            
                            grid_input_rot = (recon_from_rot_input + 1.0) / 2.0
                            grid_latent_rot = (recon_from_rot_latent + 1.0) / 2.0
                            for i in range(n_images):
                                vutils.save_image(grid_input_rot[i], f"{recon_dir}/recon_ep_{ep_print}.{fraction_print}_img_{i:02d}_rot_input_{angle_deg}.png")
                                vutils.save_image(grid_latent_rot[i], f"{recon_dir}/recon_ep_{ep_print}.{fraction_print}_img_{i:02d}_rot_latent_{angle_deg}.png")
                            
                    # Atomic system call ensures the directory is mathematically complete before removing the _tmp suffix
                    os.rename(tmp_eval_base_dir, final_eval_base_dir)
                    print(f"[RANK 0] Atomic directory rename complete. Artifacts secured.", flush=True)
                    
                    # CSV Metrics Update
                    row = {"epoch": epoch, "fraction": fraction_print, "lr": optimizer.param_groups[0]['lr']}
                    for key, v in t_stats.items(): row[f"train_{key}"] = v
                    for key, v in v_stats.items(): row[f"val_{key}"] = v
                    row["equivariance_error_25_patches"] = sum(eq_errors) / len(eq_errors)
                    
                    epoch_stats_df.append(row)
                    append_to_csv(epoch_stats_df, f"{RUN_DIR}/epoch_stats.csv")
                    
                    np.savez(f"{final_eval_base_dir}/fsq_bin_histogram_ep_{ep_print}.{fraction_print}.npz", 
                             train=train_bin_counts.cpu().numpy(), val=val_bin_counts.cpu().numpy())
                    
                    # Checkpoint Generation
                    ckpt_dir = f"{RUN_DIR}/checkpoints"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_name = f"{ckpt_dir}/checkpoint_ep_{ep_print}.{fraction_print}.pt"
                    save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, v_stats['loss_mu'], ckpt_name)
                    checkpoint_history.append(ckpt_name)
                    
                    if len(checkpoint_history) > 4:
                        os.remove(checkpoint_history.pop(0))
                        
                    if v_stats['loss_mu'] < best_val_loss:
                        best_val_loss = v_stats['loss_mu']
                        save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_val_loss, f"{ckpt_dir}/best_model.pt")

                # Synchronization Point 2: Resolves deadlocks before resuming standard training
                print(f"[RANK {local_rank}] Waiting at final evaluation barrier.", flush=True)
                dist.barrier()
                print(f"[RANK {local_rank}] Barrier resolved for Epoch {ep_print}.{fraction_print}. Resuming training.", flush=True)
                
                train_bin_counts.zero_()
                val_bin_counts.zero_()
                model.train()
                
            if CONFIG["debug_mode"] and is_endpoint:
                break
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
