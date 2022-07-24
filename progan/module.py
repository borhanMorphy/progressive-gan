from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .generator import Generator
from .discriminator import Discriminator
from .utils import compute_gradient_penalty

class ProGAN(nn.Module):
    def __init__(
        self,
        latend_dim: int = 512,
        img_chanels: int = 3,
        final_img_size: int = 1024,
        # add ons
        use_wscale: bool = True,
        use_pixelnorm: bool = True,
        use_mb_stddev: bool = True,
    ) -> None:

        self.generator = Generator(
            latend_dim=latend_dim,
            img_channels=img_chanels,
            final_img_size=final_img_size,
            use_wscale=use_wscale,
            use_pixelnorm=use_pixelnorm)

        self.discriminator = Discriminator(
            latend_dim=latend_dim,
            img_channels=img_chanels,
            final_img_size=final_img_size,
            use_wscale=use_wscale,
            use_mb_stddev=use_mb_stddev)


    def forward(self, noise: Tensor) -> Tensor:
        return self.generator(noise)

    def compute_discriminator_loss(
        self,
        x_real: Tensor,
        noise: Tensor,
        progression_step: int = 0,
        alpha: float = 1.0,
        gp_lambda: float = 10,
        eps_drift: float = 1e-3,
    ) -> Tensor:

        with torch.no_grad():
            x_fake = self.generator.progressive_forward(
                noise,
                progression_step=progression_step,
                alpha=alpha
            )

        gradient_penalty = gp_lambda * compute_gradient_penalty(x_real, x_fake, self.discriminator)
        # gradient_penalty: batch_size,

        real_scores = self.discriminator.progressive_forward(x_real, alpha=alpha)
        fake_scores = self.discriminator.progressive_forward(x_fake, alpha=alpha)

        drift_penalty = eps_drift * real_scores**2

        return (fake_scores - real_scores  + gradient_penalty + drift_penalty).mean()

    def compute_generator_loss(
        self,
        noise: Tensor,
        progression_step: int = 0,
        alpha: float = 1.0,
    ) -> Tensor:
        # TODO pydoc
        x_fake = self.generator.progressive_forward(
            noise,
            progression_step=progression_step,
            alpha=alpha
        )
        fake_scores = self.discriminator.progressive_forward(x_fake, alpha=alpha)
        return (-1 * fake_scores).mean()

    def configure_optimizers(
        self,
        learning_rate: float = 1e-3,
        betas: Tuple[float, float] = (0, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        return (
            torch.optim.Adam(
                self.generator.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay),

            torch.optim.Adam(
                self.discriminator.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay)
        )
