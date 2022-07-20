from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def compute_gradient_penalty(x_real: Tensor, x_fake: Tensor, model: nn.Module) -> Tensor:
    batch_size = x_real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=x_real.device, dtype=x_real.dtype)
    x_hat = alpha*x_real + (1-alpha) * x_fake
    x_hat.requires_grad = True

    scores = model.progressive_forward(x_hat)
    # compute gradients with respect to `x_hat`
    gradient, = torch.autograd.grad(
        inputs=x_hat,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )
    # gradient: batch_size, C, H, W

    # compute l2 norm of gradients 
    norm = gradient.flatten(start_dim=1).norm(p=2, dim=1)
    # norm: batch_size,

    return (norm - 1) ** 2

class WSConv2d(nn.Conv2d):
    """Equalized Learning Rate
    paper section: 4.1

    special thanks to;
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/ProGAN/model.py#L28

    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        fan_in = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        self.register_buffer("weight_scaler", torch.tensor(2/fan_in) ** 0.5)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input * self.weight_scaler)


class PixelWiseNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x: Tensor) -> Tensor:
        # x: B x C x H x W
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class MinibatchSTDDev(nn.Module):
    def __init__(self, groups: int = 4) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, fmap_h, fmap_w = x.shape

        groups = min(batch_size, self.groups)

        x_stats = x.reshape(groups, batch_size//groups, channels, fmap_h, fmap_w)
        # x_stats: G, M, C, H, W

        # [G x M x C x H x W] -> [M x C x H x W] -> [M x 1 x 1 x 1] -> [B x 1 x H x W]
        x_stats = x_stats.std(dim=0).mean(dim=(1, 2, 3), keepdim=True).tile((groups, 1, fmap_h, fmap_w))

        # concat along feature dim
        return torch.cat([x, x_stats], dim=1)


class Generator(nn.Module):
    default_channel_factors: Tuple[float] = (1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32)
    initial_img_size: int = 2**2
    max_img_size: int = 2**10

    conv_cls = WSConv2d
    pix_norm_cls = PixelWiseNorm

    def __init__(
        self,
        latent_dim: int = 512,
        img_channels: int = 3,
        final_img_size: int = 2**10,
        channel_factors: Tuple[float] = None,
        use_wscale: bool = True,
        use_pixelnorm: bool = True,
    ) -> None:
        super().__init__()

        assert self.initial_img_size <= final_img_size <= self.max_img_size

        if not use_wscale:
            self.conv_cls = nn.Conv2d

        if not use_pixelnorm:
            self.pix_norm_cls = nn.Identity

        num_progressive_layers = int(math.log2(final_img_size) - math.log2(self.initial_img_size))

        self.latent_dim = latent_dim

        factors = list(channel_factors or self.default_channel_factors)
        factors = factors[:num_progressive_layers]
        factors.insert(0, 1) # add initial layer

        self.to_rgb = nn.ModuleList(
            [
                self.conv_cls(
                    int(factor * latent_dim),
                    img_channels,
                    kernel_size=1,
                )
                for factor in factors
            ]
        )

        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.2),
            self.conv_cls(
                latent_dim,
                latent_dim,
                kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            self.pix_norm_cls()
        )

        self.progressive_blocks = nn.ModuleList(
            [
                self.make_progressive_block(
                    int(factors[step - 1] * latent_dim),
                    int(factors[step] * latent_dim),
                    conv_cls=self.conv_cls,
                    pix_norm_cls=self.pix_norm_cls,
                )
                for step in range(1, num_progressive_layers+1)
            ]
        )

    @staticmethod
    def make_progressive_block(in_channels: int, out_channels: int, conv_cls=nn.Conv2d, pix_norm_cls=nn.Identity) -> nn.Module:
        return nn.ModuleDict({
            "upsample": nn.UpsamplingNearest2d(scale_factor=2),
            "conv_block": nn.Sequential(
                conv_cls(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                pix_norm_cls(),
                conv_cls(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                pix_norm_cls(),
            ),
        })

    def progressive_forward(self, z: Tensor, progression_step: int = 0, alpha: float = 1.0) -> Tensor:
        x = self.initial_block(z)

        for progressive_block in self.progressive_blocks[:progression_step]:
            scaled_x = progressive_block["upsample"](x)
            x = progressive_block["conv_block"](scaled_x)

        x = self.to_rgb[progression_step](x)
        if progression_step > 0:
            prev_x = self.to_rgb[progression_step - 1](scaled_x)
            # fade in
            x = alpha * x + (1-alpha) * prev_x
        return x


    def forward(self, z: Tensor) -> Tensor:
        x = self.initial_block(z)
        for progressive_block in self.progressive_blocks:
            scaled_x = progressive_block["upsample"](x)
            x = progressive_block["conv_block"](scaled_x)

        return self.to_rgb[-1](x)

    def get_latent_sample(self, batch_size: int = 1) -> Tensor:
        # N(0,1)
        return torch.randn(batch_size, self.latent_dim, 1, 1)


class Discriminator(nn.Module):
    default_channel_factors: Tuple[float] = (1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32)
    initial_img_size: int = 2**2
    max_img_size: int = 2**10

    conv_cls = WSConv2d
    mb_stddev_cls = MinibatchSTDDev

    def __init__(
        self,
        max_channels: int = 512,
        img_channels: int = 3,
        final_img_size: int = 2**10,
        channel_factors: Tuple[float] = None,
        use_wscale: bool = True,
        use_mb_stddev: bool = True,
    ) -> None:
        super().__init__()

        assert self.initial_img_size <= final_img_size <= self.max_img_size

        if not use_wscale:
            self.conv_cls = nn.Conv2d

        if not use_mb_stddev:
            self.mb_stddev_cls = nn.Identity

        num_progressive_layers = int(math.log2(final_img_size) - math.log2(self.initial_img_size))

        self.final_img_size = final_img_size

        factors = channel_factors or self.default_channel_factors
        factors = list(reversed(factors[:num_progressive_layers]))
        factors.append(1) # add last layer factor

        self.from_rgb = nn.ModuleList(
            [
                nn.Sequential(
                    self.conv_cls(
                        img_channels,
                        int(factor * max_channels),
                        kernel_size=1,
                    ),
                    nn.LeakyReLU(negative_slope=0.2),
                )
                for factor in factors
            ]
        )

        self.progressive_blocks = nn.ModuleList(
            [
                self.make_progressive_block(
                    int(factors[step] * max_channels),
                    int(factors[step + 1] * max_channels),
                    conv_cls=self.conv_cls,
                )
                for step in range(num_progressive_layers)
            ]
        )

        self.initial_block = nn.Sequential(
            self.mb_stddev_cls(),
            self.conv_cls(
                max_channels + 1 if use_mb_stddev else 0,
                max_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            self.conv_cls(max_channels, max_channels, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(max_channels, 1)
        )

    @staticmethod
    def make_progressive_block(in_channels: int, out_channels: int, conv_cls=nn.Conv2d) -> nn.Module:
        return nn.ModuleDict({
            "conv_block": nn.Sequential(
                conv_cls(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
                conv_cls(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            "downsample": nn.AvgPool2d(kernel_size=2, stride=2),
        })

    def progressive_forward(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        img_size = x.size(-1)
        start_step = int(math.log2(self.final_img_size) - math.log2(img_size))
        # max: 8
        # min: 0
        out = self.from_rgb[start_step](x)

        for i, progressive_block in enumerate(self.progressive_blocks[start_step:]):
            out = progressive_block["conv_block"](out)
            out = progressive_block["downsample"](out)
            if i == 0:
                prev_out = self.from_rgb[start_step + 1](
                    F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
                )
                # fade in
                out = alpha * out + (1 - alpha) * prev_out

        return self.initial_block(out)


    def forward(self, x: Tensor) -> Tensor:
        out = self.from_rgb[-1](x)

        for progressive_block in self.progressive_blocks:
            out = progressive_block["conv_block"](out)
            out = progressive_block["downsample"](out)

        return self.initial_block(out)
