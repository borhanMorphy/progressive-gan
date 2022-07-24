from typing import Tuple
import math

import torch
import torch.nn as nn
from torch import Tensor

from .add_ons import WSConv2d, PixelWiseNorm

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
