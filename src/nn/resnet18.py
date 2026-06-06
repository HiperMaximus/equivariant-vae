from abc import ABC, abstractmethod
from typing import (
    Literal,
    ParamSpec,
    TypeVar,
    override,
)

import torch
from torch import Tensor, nn

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


class Stem(BaseModule[[Tensor], Tensor]):
    def __init__(self) -> None:
        super().__init__()
        # 1. Conv1: 7x7, 64 filters, stride 2
        # Padding=3 is calculated to halve dimensions: (H + 2*3 - 7)/2 + 1
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # 2. Batch Normalization: "right after each convolution"
        self.bn = nn.BatchNorm2d(64)

        # 3. Activation: ReLU
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self) -> None:
        # He/Kaiming Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        # Initialize BN parameters
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicBlock(BaseModule[[Tensor], Tensor]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Literal[1, 2] = 1,
    ) -> None:
        super().__init__()

        # --- Main Path ---
        # First conv handles the stride (downsampling)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second conv is always stride 1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Shortcut Path ---
        # If dimensions change (stride > 1 or channels differ), we need a projection.
        # This corresponds to Eqn.(2) and Option B in the paper.
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.init_weights()

    def init_weights(self) -> None:
        # Standard initialization for ResNet layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply the shortcut projection if necessary
        if self.downsample is not None:
            identity = self.downsample(x)

        # Element-wise addition (Resnet skip connection)
        out += identity

        # "We adopt the second nonlinearity after the addition"
        out = self.relu(out)

        return out


class ResNet18Encoder(BaseModule[[Tensor], tuple[Tensor, Tensor]]):
    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()

        # 1. The Stem (Initial Downsampling)
        # Input: (B, 3, H, W) -> Output: (B, 64, H/2, W/2)
        self.stem = Stem()

        # 2. The Residual Layers (Stages)
        # Each layer is a sequence of BasicBlocks.

        # conv2_x: 64 filters, 2 blocks, stride 1
        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            blocks=2,
            stride=1,
        )

        # conv3_x: 128 filters, 2 blocks, stride 2
        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            blocks=2,
            stride=2,
        )

        # conv4_x: 256 filters, 2 blocks, stride 2
        self.layer3 = self._make_layer(
            in_channels=128,
            out_channels=256,
            blocks=2,
            stride=2,
        )

        # conv5_x: 512 filters, 2 blocks, stride 1
        self.layer4 = self._make_layer(
            in_channels=256,
            out_channels=512,
            blocks=2,
            stride=1,
        )
        # --- The Gaussian Head ---
        # 1. Normalization (GroupNorm is standard for VAE bottlenecks)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=512)

        # 2. Activation
        self.act_out = nn.SiLU(inplace=True)

        # 3. Projection to (Mean, LogVar)
        # Output channels = 2 * latent_dim (e.g., 8 channels for dim 4)
        self.conv_out = nn.Conv2d(512, 2 * latent_dim, kernel_size=3, padding=1)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: Literal[1, 2],
    ) -> nn.Sequential:
        """Help building a stage containing a sequence of BasicBlocks."""
        layers: list[nn.Module] = []

        # The first block in a stage handles the downsampling (stride)
        # and channel expansion.
        layers.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            ),
        )

        # Subsequent blocks in the stage always have stride=1
        # and input_channels == output_channels.
        layers.extend(
            BasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
            )
            for _ in range(1, blocks)
        )

        return nn.Sequential(*layers)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z from the Gaussian distribution N(mu, std) using the reparameterization trick.

        z = mu + std * epsilon.
        """
        # Standard training mode
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Stem
        x = self.stem(x)

        # Convolutional Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.norm_out(x)
        x = self.act_out(x)
        moments = self.conv_out(x)

        # Split into Mean and LogVariance
        mu, logvar = moments.chunk(2, dim=1)

        return mu, logvar


class ResNet18Decoder(BaseModule[[Tensor], Tensor]):
    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()

        # 1. Initial Projection
        # Input: (B, latent_dim, H/8, W/8) -> (B, 512, H/8, W/8)
        self.conv_in = nn.Conv2d(
            in_channels=latent_dim,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 2. Residual Stages (Reverse of Encoder)

        # Layer 4 Inverse: 512 -> 256 channels.
        # Encoder Layer 4 was Stride 1 (32x32), so Decoder maintains size here.
        self.layer4 = self._make_layer(512, 256, blocks=2)

        # Upsample 1: 32x32 -> 64x64
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        # Layer 3 Inverse: 256 -> 128 channels.
        self.layer3 = self._make_layer(256, 128, blocks=2)

        # Upsample 2: 64x64 -> 128x128
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        # Layer 2 Inverse: 128 -> 64 channels.
        self.layer2 = self._make_layer(128, 64, blocks=2)

        # Upsample 3: 128x128 -> 256x256 (Replaces Stem downsampling)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        # Layer 1 Inverse: 64 -> 64 channels.
        self.layer1 = self._make_layer(64, 64, blocks=2)

        # 3. Output Head
        # Standard VAE Output: Norm -> SiLU -> Conv to RGB
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=64)
        self.act_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.init_weights()
        # Apply Zero Initialization specifically to the output head
        nn.init.constant_(self.conv_out.weight, 0)
        if self.conv_out.bias is not None:
            nn.init.constant_(self.conv_out.bias, 0)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
    ) -> nn.Sequential:
        """Build a decoder stage.
        Since we handle upsampling separately with nn.Upsample, these blocks
        just handle channel reduction and feature processing with stride=1.
        """
        layers: list[nn.Module] = []

        # The first block handles the channel change (e.g., 512 -> 256).
        # BasicBlock automatically adds a 1x1 projection on the shortcut
        # if in_channels != out_channels.
        layers.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
            ),
        )

        # Subsequent blocks
        layers.extend(
            BasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
            )
            for _ in range(1, blocks)
        )

        return nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Initial projection
        x = self.conv_in(x)

        # Stage 4 (Inverse)
        x = self.layer4(x)

        # Stage 3 (Inverse)
        x = self.up1(x)
        x = self.layer3(x)

        # Stage 2 (Inverse)
        x = self.up2(x)
        x = self.layer2(x)

        # Stage 1 (Inverse)
        x = self.up3(x)
        x = self.layer1(x)

        # Output
        x = self.norm_out(x)
        x = self.act_out(x)
        return self.conv_out(x)


class ResNet18VAE(BaseModule[[Tensor], tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()
        self.encoder = ResNet18Encoder(latent_dim=latent_dim)
        self.decoder = ResNet18Decoder(latent_dim=latent_dim)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Standard VAE forward pass."""
        mu, logvar = self.encoder(x)
        z = ResNet18Encoder.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        """Decode latents directly.

        Needed for EQ-VAE to decode transformed latents: D(tau(z)).
        """
        return self.decoder(z)


# --- Verification Script ---


if __name__ == "__main__":
    print("--- Starting ResNet18VAE Verification ---")

    # 1. Setup
    HEIGHT, WIDTH = 256, 256
    LATENT_DIM = 4
    BATCH_SIZE = 16

    # Create random batch
    test_tensor = torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH)
    print(f"Input Shape: {test_tensor.shape}")

    # Initialize Model
    model = ResNet18VAE(latent_dim=LATENT_DIM)

    # 2. Forward Pass
    try:
        recon_x, mu, logvar = model(test_tensor)

        # 3. Validation Prints
        print("\n--- Output Shapes ---")
        print(f"Reconstruction: {recon_x.shape}")
        print(f"Mu (Mean):      {mu.shape}")
        print(f"LogVar:         {logvar.shape}")

        # 4. Correctness Checks
        # Check Spatial Dimensions (Should correspond to f=8 compression)
        # 256 / 8 = 32
        assert mu.shape == (BATCH_SIZE, LATENT_DIM, 32, 32), (
            f"ERROR: Latent shape mismatch. Expected (B, {LATENT_DIM}, 32, 32), got {mu.shape}"
        )

        assert recon_x.shape == test_tensor.shape, (
            f"ERROR: Output shape mismatch. Expected {test_tensor.shape}, got {recon_x.shape}"
        )

        # Check for NaNs
        has_nan = (
            torch.isnan(recon_x).any()
            or torch.isnan(mu).any()
            or torch.isnan(logvar).any()
        )
        if has_nan:
            print("FAILURE: NaN values detected in output!")
        else:
            print("SUCCESS: No NaNs detected.")
            print("SUCCESS: Shapes are correct.")

    except Exception as e:
        print(f"CRASH: Model forward pass failed with error: {e}")
