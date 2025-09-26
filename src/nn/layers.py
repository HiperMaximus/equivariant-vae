"""Normalization-Free Lie Group Variational Autoencoder (LGAE).

This module implements a Variational Autoencoder whose encoder outputs Lie-algebra
vectors that are exponentiated into UTDAT (upper-triangular positive definite affine
transform) elements to parameterize diagonal-Gaussian latents. The design is
normalization-free and uses stability tricks (Fixup/SkipInit/ReZero + orthogonal or
delta-orthogonal init) so training is well-behaved without BatchNorm.

Key ideas
- Encoder outputs a Lie-algebra vector v; expmap(v) produces a UTDAT element that
  encodes (μ, σ) for a diagonal Gaussian. Sampling is then z = μ + σ ⊙ ε.
- Decoder defines the observation model p(x|z). Reconstruction is the negative
  log-likelihood (NLL) under a chosen likelihood (Gaussian/Laplace/Bernoulli).
- Latent regularization can be standard KL(q(z|x) || p(z)) OR an “intrinsic” penalty
  in the Lie algebra (‖v‖²), which approximates the geodesic distance to the identity.
- Normalization-free stability: residual branches are down-scaled (Fixup/SkipInit) and
  optionally gated with ReZero α (initialized at 0), with orthogonal or delta-orthogonal
  kernel init to preserve signal propagation at depth.

Training tips
- Start with Gaussian recon loss with a learned global log-variance (stable & simple).
- β-VAE anneal: linearly or sigmoid-ramp β from 0 → β_max over 10–30% warmup.
- Free-bits: clamp per-dim KL to a small τ (e.g., 0.5–1.0 nats) to prevent posterior
  collapse early on.
- Optimizer: AdamW, optional Adaptive Gradient Clipping (AGC) for large models.

References
----------
- Lie Group Auto-Encoder (UTDAT, expmap, intrinsic loss): Gong & Cheng, 2019.
- BatchNorm biases residual blocks toward identity / SkipInit: De & Smith, 2020.
- ReZero residual gating: Bachlechner et al., 2020.
- Delta-Orthogonal init for deep CNNs: Xiao et al., 2018.
- Normalizer-Free ResNets & AGC: Brock et al., 2021.

"""  # noqa: RUF002

import math
from abc import ABC, abstractmethod
from typing import Literal, ParamSpec, TypeVar, override

import torch
import torch.nn.functional as F  # noqa: N812
from pytorch_msssim import MS_SSIM  # pyright: ignore[reportMissingTypeStubs]
from torch import Tensor, nn


def fixup_scale(L: int, m: int) -> float:  # noqa: N803
    """Compute Fixup scaling for residual branches.

    Args:
        L: total number of residual blocks in the network.
        m: number of linear layers in the residual branch.

    Returns:
        scale factor to multiply weights in the residual branch.

    """
    min_m = 2  # Min number of linear layers in res branch for fixup
    return L ** (-1.0 / (2 * m - 2)) if m >= min_m else L ** (-0.5)


def init_conv(weight: Tensor) -> None:
    """Initialize conv layers filling with appropiate weights.

    Args:
        weight: weight tensor to initialize - shape (C_out, C_in, kH, kW)

    """
    with torch.no_grad():
        C_out, C_in, kH, kW = weight.shape  # noqa: N806
        m, n = C_out, C_in * kH * kW
        if m > n:
            torch.nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="linear")
        else:
            torch.nn.init.orthogonal_(weight)


Out_co = TypeVar("Out_co", covariant=True)
P_in = ParamSpec("P_in")


class BaseModule[**P_in, Out_co](nn.Module, ABC):
    """Module class to inherit from, so all modules satisfy type checking.

    Silences Pylance(reportUnknownMemberType) when calling super().__init__() needed for all layers.
    Gives correct type for __call__() to propagate input/output types.
    Also enforces that subclasses implement forward() with correct return type.
    Meaning:
    - Calling a subclass instance returns `Out_co`.
    - Subclasses' `forward` are checked to return `Out_co`.
    """

    def __init__(self) -> None:
        """Init that calls the parent nn.Module init method.

        With '# type: ignore[reportUnknownMemberType]' allows type checking to work properly.
        """
        super().__init__()  # type: ignore[reportUnknownMemberType]

    # Calling a module with inputs P_in and returns Out_co
    @override
    def __call__(
        self,
        *args: P_in.args,
        **kwargs: P_in.kwargs,
    ) -> Out_co:  # narrowing from the default Pytorch Any is fine
        """Call that keeps the correct type for input/output.

        Args:
            *args: positional arguments for forward.
            **kwargs: keyword arguments for forward.

        Returns:
            Output of forward with type Out_co.

        """
        return super().__call__(*args, **kwargs)  # keep hooks/autocast/compile path

    # Forward should also inputs P_in and return Out_co, so subclasses are checked against it
    @override
    @abstractmethod
    def forward(self, *args: P_in.args, **kwargs: P_in.kwargs) -> Out_co:
        """Forward method to be implemented by subclasses."""
        ...


class StemEncoder(BaseModule[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding="same",
            bias=True,
        )

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.conv.weight)
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SiLU_bias(BaseModule[[Tensor], Tensor]):
    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.silu(x) + self.bias


class PointWiseConv(BaseModule[[Tensor], Tensor]):
    conv: nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nonlinearity: Literal["linear", "silu"] = "linear",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights(nonlinearity=nonlinearity)

    def init_weights(self, nonlinearity: Literal["linear", "silu"] = "linear") -> None:
        with torch.no_grad():
            init_conv(self.conv.weight)
            if nonlinearity == "silu":
                self.conv.weight.mul_(1.676)  # Correct gain for SiLU
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DepthWiseConv(BaseModule[[Tensor], Tensor]):
    conv: nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        nonlinearity: Literal["linear", "silu"] = "linear",
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=in_channels,
            bias=True,
        )

        self.init_weights(nonlinearity=nonlinearity)

    def init_weights(self, nonlinearity: Literal["linear", "silu"] = "linear") -> None:
        with torch.no_grad():
            init_conv(self.conv.weight)
            if nonlinearity == "silu":
                self.conv.weight.mul_(1.676)  # Correct gain for SiLU
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SqueezeExcite(BaseModule[[Tensor], Tensor]):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = PointWiseConv(
            in_channels=channels,
            out_channels=reduced_channels,
            nonlinearity="silu",
        )
        self.act = nn.SiLU()
        self.fc2 = PointWiseConv(
            in_channels=reduced_channels,
            out_channels=channels,
            nonlinearity="linear",
        )
        self.sigmoid = nn.Sigmoid()
        self.alpha_gate = nn.Parameter(torch.ones(1))
        self.beta_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            self.fc2.conv.weight.zero_()
            self.alpha_gate.fill_(2.0)  # Start with no effect

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.alpha_gate * self.sigmoid(y) + self.beta_bias
        return x * y


class ResBlock(BaseModule[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        expansion_factor: int = 2,
        kernel_size: int = 3,
        L: int = 6,
    ) -> None:
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.initial_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.alpha_gate = nn.Parameter(torch.zeros(1))
        self.pw1 = PointWiseConv(
            in_channels=in_channels,
            out_channels=expanded_channels,
            nonlinearity="silu",
        )
        self.act1 = SiLU_bias(channels=expanded_channels)
        self.dw = DepthWiseConv(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=1,
            nonlinearity="silu",
        )
        self.act2 = SiLU_bias(channels=expanded_channels)
        self.se = SqueezeExcite(channels=expanded_channels)
        self.pw2 = PointWiseConv(
            in_channels=expanded_channels,
            out_channels=in_channels,
        )  # Linear gain

        self.init_weights(L=L)

    def init_weights(self, L: int) -> None:
        m = 3  # number of linear layers with weights
        scale = fixup_scale(L, m)

        with torch.no_grad():
            # Scale weights
            self.pw1.conv.weight.mul_(scale)
            self.dw.conv.weight.mul_(scale)
            self.pw2.conv.weight.mul_(scale)

    def forward(self, x: Tensor) -> Tensor:
        res = self.pw1(x + self.initial_bias)
        res = self.act1(res)
        res = self.dw(res)
        res = self.act2(res)
        res = self.se(res)
        res = self.pw2(res)
        return x + self.alpha_gate * res


class DownsampleBlock(BaseModule[[Tensor], Tensor]):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            PointWiseConv(in_channels=in_channels, out_channels=out_channels),
            DepthWiseConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class DownsampleStage(BaseModule[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
        L: int = 6,
    ) -> None:
        super().__init__()
        blocks: list[ResBlock] = [
            ResBlock(in_channels=in_channels, kernel_size=kernel_size, L=L)
            for _ in range(num_blocks)
        ]
        self.seq = nn.Sequential(
            *blocks,
            DownsampleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class UpsampleBlock(BaseModule[[Tensor], Tensor]):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            PointWiseConv(in_channels, in_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            DepthWiseConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
            ),
            PointWiseConv(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class UpsampleStage(BaseModule[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
        L: int = 6,
    ) -> None:
        super().__init__()
        blocks: list[ResBlock] = [
            ResBlock(in_channels=out_channels, kernel_size=kernel_size, L=L)
            for _ in range(num_blocks)
        ]
        self.seq = nn.Sequential(
            UpsampleBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            *blocks,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class FinalConvLinearHead(BaseModule[[Tensor], Tensor]):
    def __init__(self, in_channels: int, out_channels: int = 3, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            self.conv.weight.zero_()
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Encoder(BaseModule[[Tensor], tuple[Tensor, Tensor]]):
    def __init__(self) -> None:
        super().__init__()
        L = 6
        num_blocks = 2
        self.stem = StemEncoder()  # 32x32x3 -> 32x32x16
        self.stage1 = DownsampleStage(
            in_channels=16,
            out_channels=32,
            num_blocks=num_blocks,
            kernel_size=5,
            L=L,
        )  # 32x32x16 -> 16x16x32
        self.stage2 = DownsampleStage(
            in_channels=32,
            out_channels=64,
            num_blocks=num_blocks,
            L=L,
        )  # 16x16x32 -> 8x8x64
        self.stage3 = DownsampleStage(
            in_channels=64,
            out_channels=128,
            num_blocks=num_blocks,
            L=L,
        )  # 8x8x64 -> 4x4x128
        self.theta_head = PointWiseConv(in_channels=128, out_channels=24)  # 4x4x128 -> 4x4x24
        self.phi_head = PointWiseConv(in_channels=128, out_channels=24)  # 4x4x128 -> 4x4x24
        # compression factor: (32*32*3)/(4*4*24) = 8

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        theta = self.theta_head(x)
        phi = self.phi_head(x)
        return theta, phi


class Decoder(BaseModule[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        L = 6
        num_blocks = 2
        self.initial_pw = PointWiseConv(in_channels=24, out_channels=128)  # 4x4x24 -> 4x4x128
        self.stage1 = UpsampleStage(
            in_channels=128,
            out_channels=64,
            num_blocks=num_blocks,
            L=L,
        )  # 4x4x128 -> 8x8x64
        self.stage2 = UpsampleStage(
            in_channels=64,
            out_channels=32,
            num_blocks=num_blocks,
            kernel_size=5,
            L=L,
        )  # 8x8x64 -> 16x16x32
        self.stage3 = UpsampleStage(
            in_channels=32,
            out_channels=16,
            num_blocks=num_blocks,
            kernel_size=5,
            L=L,
        )  # 16x16x32 -> 32x32x16
        self.final_conv = FinalConvLinearHead(in_channels=16, out_channels=3)  # 32x32x16 -> 32x32x3

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial_pw(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.final_conv(x)


class PPSFFT2(BaseModule[[Tensor], Tensor]):
    """P+S decomposition in Fourier domain using real FFT to avoid leakage.

    forward(u) -> P_hat  where P_hat = rfft2(u) - B_hat/λ
    Input:  (B, C, H, W) real
    Output: (B, C, H, W//2 + 1) complex
    """

    # type checker hint
    lam: Tensor
    w_top: Tensor
    w_bot: Tensor

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        norm: Literal["ortho", "backward", "forward"] | None = "ortho",
        eta: float = 0.375,
        rho: float = 1 / 3,
    ) -> None:
        super().__init__()
        self.H, self.W = height, width
        self.norm = norm
        lam = self._laplacian_eigs_rfft2()
        self.register_buffer("lam", lam.view(1, 1, *lam.shape), persistent=False)

        # --- choose radii from size ---
        if not (0 < eta <= 1):
            msg = "eta must be in (0, 1]."
            raise ValueError(msg)
        if not (0 <= rho < 1):
            msg = "rho must be in [0, 1)."
            raise ValueError(msg)

        nyq = min(self.H, self.W) // 2  # Nyquist radius (in bins)
        r1 = max(1, min(math.floor(eta * nyq), nyq))
        r0 = max(0, min(round(rho * r1), r1 - 1))  # ensure 0 ≤ r0 < r1

        # small low-freq weighting patches
        w_top, w_bot = self.build_lowfreq_patches_rfft2(r0=r0, r1=r1)
        self.register_buffer("w_top", w_top, persistent=False)
        self.register_buffer("w_bot", w_bot, persistent=False)

        self.r0, self.r1 = r0, r1

    def _laplacian_eigs_rfft2(self) -> Tensor:
        ky = 2 * torch.pi * torch.arange(self.H).view(self.H, 1) / self.H
        kx = 2 * torch.pi * torch.arange(self.W // 2 + 1).view(1, self.W // 2 + 1) / self.W
        # Paper denominator: lambda = 2 cos(kx) + 2 cos(ky) - 4   (<= 0, with D[0,0]=0)
        lam: Tensor = 2.0 * (torch.cos(kx) + torch.cos(ky) - 2.0)  # (H, W//2+1)
        lam[0, 0] = -torch.inf  # enforce ŝ(0,0)=0
        return lam

    def build_lowfreq_patches_rfft2(self, r0: int = 1, r1: int = 3) -> tuple[Tensor, Tensor]:
        """Return (w_top, w_bot) weighting patches for low-freq Fourier coeffs.

        Args:
            r0: inner radius of cosine ramp (DC floor)
            r1: outer radius of cosine ramp (full pass)

        Returns:
            w_top: (r1+1, r1+1) for rows [0:r1+1], cols [0:r1+1]
            w_bot: (r1,   r1+1) for rows [H-r1:H], cols [0:r1+1]
        Only these entries are < 1; everything else is implicitly 1.

        Raises:
            ValueError: if not (0 <= r0 < r1 <= min(H/2, W/2))

        """
        r1 = min(int(r1), self.H // 2, self.W // 2)
        if not (0 <= r0 < r1):
            msg = f"""Require 0 <= r0 < r1 <= min(H/2, W/2);
            got r0={r0}, r1={r1}, H={self.H}, W={self.W}"""
            raise ValueError(msg)

        device = self.lam.device  # or self.parameters().__next__().device
        dtype = self.lam.dtype  # keep it consistent with lam / FFT dtype

        # common x bins (0..r1)
        ix = torch.arange(0, r1 + 1, device=device, dtype=dtype).view(1, -1)

        # ---- top block (dy = 0..r1)
        iy_top = torch.arange(0, r1 + 1, device=device, dtype=dtype).view(-1, 1)
        r_top = torch.sqrt(iy_top * iy_top + ix * ix)
        w_top = torch.ones_like(r_top)
        w_top[r_top <= r0] = 0.0  # DC floor
        in_ramp = (r_top > r0) & (r_top < r1)
        w_top[in_ramp] = 0.5 * (1.0 - torch.cos(math.pi * (r_top[in_ramp] - r0) / (r1 - r0)))
        # enforce exact DC:
        w_top[0, 0] = 0.0

        # ---- bottom block (dy = 1..r1 across wrap)
        iy_bot = torch.arange(1, r1 + 1, device=device, dtype=dtype).view(-1, 1)
        r_bot = torch.sqrt(iy_bot * iy_bot + ix * ix)  # shape (r1, r1+1)
        w_bot = torch.ones_like(r_bot)
        w_bot[r_bot <= r0] = 0.0  # DC floor
        in_ramp = (r_bot > r0) & (r_bot < r1)
        w_bot[in_ramp] = 0.5 * (1.0 - torch.cos(math.pi * (r_bot[in_ramp] - r0) / (r1 - r0)))

        return w_top, w_bot  # small patches only

    def rfft_typed(self, x: Tensor) -> Tensor:
        return torch.fft.rfft2(x, norm=self.norm)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    def forward(self, u: Tensor) -> Tensor:
        # 1) Border image b encodes opposite-side mismatches
        b = torch.zeros_like(u)
        d_tb = u[..., -1, :] - u[..., 0, :]  # top/bottom gap
        b[..., 0, :] += d_tb
        b[..., -1, :] -= d_tb
        d_lr = u[..., :, -1] - u[..., :, 0]  # left/right gap
        b[..., :, 0] += d_lr
        b[..., :, -1] -= d_lr

        # 2) FFTs
        B_hat = self.rfft_typed(b)
        U_hat = self.rfft_typed(u)

        # 3) Poisson solve in Fourier: ŝ = b̂ / λ  (broadcast over B,C)
        S_hat = B_hat / self.lam

        # 4) Periodic component
        P_hat = U_hat - S_hat  # p̂ = û - ŝ
        return P_hat


class ScharrGrad(BaseModule[[Tensor], tuple[Tensor, Tensor]]):
    """Scharr gradients per channel via grouped conv.

    norm='l1' -> divide by sum(|coeffs|)  (Sobel: 8, Scharr: 32)
    norm='l2' -> divide by sqrt(sum(coeffs^2))
    norm=None -> no scaling (raw integer coefficients).
    """

    kx: Tensor
    ky: Tensor

    def __init__(self, channels: int = 1, norm: Literal["l1", "l2"] | None = "l1") -> None:
        super().__init__()
        kx = torch.tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]])
        ky = kx.t()

        if norm == "l1":
            scale = kx.abs().sum()  # 32.0
            kx, ky = kx / scale, ky / scale
        elif norm == "l2":
            scale = torch.sqrt((kx**2).sum())  # unit-energy
            kx, ky = kx / scale, ky / scale

        self.channels = channels
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # padding="valid" to avoid boundary effects
        im_x = F.conv2d(x, self.kx, groups=self.channels)
        im_y = F.conv2d(x, self.ky, groups=self.channels)
        return im_x, im_y


class LumaY(BaseModule[[Tensor], Tensor]):
    w_luma: Tensor  # type checker hint

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "w_luma",
            Tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:  # x: (B,3,H,W) linear RGB
        """Convert *linear* RGB to luminance (Rec.709/sRGB primaries).

        Args:
            x: (B,3,H,W), float in [0,1] and already linearized (not sRGB).

        Returns:
            grayscale: (B,1,H,W).

        """
        return (x * self.w_luma).sum(dim=1, keepdim=True)


class LGAE(BaseModule[[Tensor, float], tuple[Tensor, Tensor, Tensor]]):
    expm1_threshold: Tensor  # type checker hint

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.to_grayscale = LumaY()
        self.ppsfft2 = PPSFFT2()
        self.mssim = MS_SSIM(data_range=1.0, channel=1, size_average=True)
        self.scharr = ScharrGrad(channels=1, norm="l1")

        # Threshold for stable expm1_over_x in float16 (half precision)
        float16_valid_threshold = 1e-2
        self.register_buffer("expm1_threshold", torch.as_tensor(float16_valid_threshold))

    def expm1_over_x_stable(
        self,
        phi: Tensor,
        clamp_min: float = -9.0,
        clamp_max: float = 9.0,
    ) -> Tensor:
        """Calculate expmp1(x)/x in a numerically stable way.

        Args:
            phi (Tensor): var part of lie algebra vector.
            clamp_min (float): min value to avoid 0/NaN.
            clamp_max (float): max value to avoid 0/NaN

        Returns:
            out (Tensor): expmp1(x)/x in a numerically stable way.

        """
        # Let AMP/autocast pick dtype; clamp to keep exp stable in half precision
        thresh = self.expm1_threshold
        x = phi.clamp(
            clamp_min,
            clamp_max,
        )  # we can drop this since we are already clamping in forward
        t = x.abs()

        # allocate result (only compute the risky branch where needed)
        out = torch.empty_like(x)
        small = t <= thresh
        large = ~small

        if small.any():
            xs = x[small]
            # Use Taylor expansion up to x^3
            out[small] = 1 + xs * (0.5 + xs * ((1 / 6) + xs * (1 / 24)))

        if large.any():
            xl = x[large]
            out[large] = torch.expm1(xl) / xl

        return out

    def intrinsic_loss_freebits(
        self,
        theta: Tensor,
        phi: Tensor,
        tau_dim: float = 0.015,
        tau_px: float = 0.05,
        beta: float = 0.5,
    ) -> Tensor:
        R: Tensor = theta.pow(2) + phi.pow(2)  # [B,C,H,W]
        # 1) per-dim floor
        R_d = R.clamp(min=tau_dim)  # [B,C,H,W]

        # 2) per-pixel floor (on channel-sum)
        _, C, _, _ = R.shape
        tau_px_floor = tau_px + tau_dim * C  # total per-pixel floor
        R_px = R_d.sum(dim=1)  # [B,H,W]
        R_hybrid = R_px.clamp(min=tau_px_floor)  # [B,H,W]

        return beta * R_hybrid.mean()

    def fourier_loss(
        self,
        x: Tensor,
        recon_x: Tensor,
    ) -> Tensor:
        P_hat_x = self.ppsfft2(x)
        P_hat_recon_x = self.ppsfft2(recon_x)
        Err: Tensor = P_hat_recon_x - P_hat_x  # error in Fourier domain
        _, _, H, _ = Err.shape
        r1 = self.ppsfft2.w_top.shape[0] - 1
        # top block
        Err[..., 0 : r1 + 1, 0 : r1 + 1] *= self.ppsfft2.w_top.view(1, 1, r1 + 1, r1 + 1)
        # bottom block (wrap ky = -1..-r1)
        Err[..., H - r1 : H, 0 : r1 + 1] *= self.ppsfft2.w_bot.view(1, 1, r1, r1 + 1)

        return Err.abs().mean()

    def edge_loss(
        self,
        x: Tensor,
        recon_x: Tensor,
    ) -> Tensor:
        imx, imy = self.scharr(x)
        recon_imx, recon_imy = self.scharr(recon_x)
        dx_loss = F.l1_loss(recon_imx, imx)
        dy_loss = F.l1_loss(recon_imy, imy)
        return 0.5 * (dx_loss + dy_loss)

    def reconstruction_loss(
        self,
        x: Tensor,
        recon_x: Tensor,
        pixel_loss_weight: float = 1.0,
        structural_loss_weight: float = 0.25,
        edge_loss_weight: float = 0.1,
        fourier_loss_weight: float = 0.1,
    ) -> Tensor:
        # range x, recon_x in [0,1]
        pixel_loss = F.l1_loss(recon_x, x)

        # convert to grayscale
        y = self.to_grayscale(x)
        recon_y = self.to_grayscale(recon_x)

        structural_loss = 1.0 - self.mssim(y, recon_y)
        edge_loss = self.edge_loss(y, recon_y)
        fourier_loss = self.fourier_loss(y, recon_y)
        return (
            pixel_loss_weight * pixel_loss
            + structural_loss_weight * structural_loss
            + edge_loss_weight * edge_loss
            + fourier_loss_weight * fourier_loss
        )

    def forward(
        self,
        x: Tensor,
        noise_scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        theta, phi = self.encoder(x)  # both [B, C, H, W]
        phi = torch.clamp(input=phi, min=-9.0, max=9.0)  # avoid NaNs
        sigma = torch.exp(phi)  # [B, C, H, W]
        scale = self.expm1_over_x_stable(phi)  # stable
        mu = theta * scale  # [B, C, H, W]
        eps = torch.randn_like(mu)
        z = mu + noise_scale * sigma * eps  # [B, C, H, W]
        recon_x = self.decoder(z)
        return recon_x, theta, phi  # (return phi/theta, not mu/logvar)
