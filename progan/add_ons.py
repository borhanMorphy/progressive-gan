import torch
import torch.nn as nn
from torch import Tensor

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
        if batch_size % groups:
            groups = batch_size

        x_stats = x.reshape(groups, batch_size//groups, channels, fmap_h, fmap_w)
        # x_stats: G, M, C, H, W

        # [G x M x C x H x W] -> [M x C x H x W] -> [M x 1 x 1 x 1] -> [B x 1 x H x W]
        x_stats = x_stats.std(dim=0).mean(dim=(1, 2, 3), keepdim=True).tile((groups, 1, fmap_h, fmap_w))

        # concat along feature dim
        return torch.cat([x, x_stats], dim=1)
