from typing import Tuple, Dict
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T

from .generator import Generator
from .discriminator import Discriminator
from .utils import compute_gradient_penalty, SquarePad
from .config import ModelConfig


class ProGAN(nn.Module):

    initial_img_size: int = 2**2

    def __init__(
        self,
        latent_dim: int = 512,
        img_channels: int = 3,
        final_img_size: int = 1024,
        # add ons
        use_wscale: bool = True,
        use_pixelnorm: bool = True,
        use_mb_stddev: bool = True,
    ) -> None:
        super().__init__()

        num_progressive_layers = int(math.log2(final_img_size / self.initial_img_size))

        self.config = dict(
            latent_dim=latent_dim,
            img_channels=img_channels,
            final_img_size=final_img_size,
            use_wscale=use_wscale,
            use_pixelnorm=use_pixelnorm,
            use_mb_stddev=use_mb_stddev,
        )

        self.generator = Generator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            num_progressive_layers=num_progressive_layers,
            use_wscale=use_wscale,
            use_pixelnorm=use_pixelnorm)

        self.discriminator = Discriminator(
            max_channels=latent_dim,
            img_channels=img_channels,
            num_progressive_layers=num_progressive_layers,
            use_wscale=use_wscale,
            use_mb_stddev=use_mb_stddev)

    @property
    def latent_dim(self) -> int:
        return self.config["latent_dim"]

    @property
    def img_channels(self) -> int:
        return self.config["img_channels"]

    @property
    def num_progression(self) -> int:
        return int(math.log2(self.config["final_img_size"] / self.initial_img_size))

    def state_dict(self) -> Dict:
        return dict(
            weights=super().state_dict(),
            config=self.config,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model = cls.from_config(
            ModelConfig(**ckpt["config"])
        )

        model.load_state_dict(
            OrderedDict([
                # to fix name change while using pytorch_lightnint
                (key.replace("_module.", ""), value)
                for key, value in ckpt["weights"].items()
            ])
        )

        return model

    @classmethod
    def from_config(cls, config: ModelConfig):
        return cls(**config.dict())

    @torch.no_grad()
    def forward(self, noise: Tensor, progression_step: int = None, alpha: float = 1.0) -> Tensor:
        progression_step = self.num_progression if progression_step is None else progression_step
        fake_imgs = self.generator(
            noise,
            progression_step=progression_step,
            alpha=alpha
        )
        return ((fake_imgs + 1) / 2).clip(min=0, max=1)

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
            x_fake = self.generator(
                noise,
                progression_step=progression_step,
                alpha=alpha)

        gradient_penalty = gp_lambda * compute_gradient_penalty(
            x_real,
            x_fake,
            self.discriminator,
            progression_step=progression_step)
        # gradient_penalty: batch_size,

        real_scores = self.discriminator(
            x_real,
            progression_step=progression_step,
            alpha=alpha)

        fake_scores = self.discriminator(
            x_fake,
            progression_step=progression_step,
            alpha=alpha)

        drift_penalty = eps_drift * real_scores**2

        return (fake_scores - real_scores  + gradient_penalty + drift_penalty).mean()

    def compute_generator_loss(
        self,
        noise: Tensor,
        progression_step: int = 0,
        alpha: float = 1.0,
    ) -> Tensor:
        x_fake = self.generator(
            noise,
            progression_step=progression_step,
            alpha=alpha
        )

        fake_scores = self.discriminator(
            x_fake,
            progression_step=progression_step,
            alpha=alpha)

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

    def initialize_weights(self):
        for layer in self.generator.modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            nn.init.constant_(layer.bias, 0)
            nn.init.normal_(layer.weight, mean=0, std=1)
            
        for layer in self.discriminator.modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            nn.init.constant_(layer.bias, 0)
            nn.init.normal_(layer.weight, mean=0, std=1)

    def get_image_size(self, progression_step: int = 0) -> int:
        return int(self.initial_img_size * 2**progression_step)

    @staticmethod
    def get_transform(img_size: int, mean: float = 0.5, std: float = 0.5):
        return T.Compose(
            [
                SquarePad(),
                T.ToTensor(),
                T.Resize(img_size),
                T.Normalize(mean, std),
            ]
        )
