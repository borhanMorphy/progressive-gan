from typing import Tuple
import math

from torch import Tensor
import torch.nn as nn


from .add_ons import WSConv2d, MinibatchSTDDev


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
