from collections import Counter

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.lite import LightningLite

from torchvision.datasets.vision import VisionDataset

from .config import TrainerConfig
from .utils import generate_noise

class LiteTrainer(LightningLite):

    @property
    def logger(self) -> SummaryWriter:
        if self._logger is None:
            self.logger = SummaryWriter()
        return self._logger

    @logger.setter
    def logger(self, logger: SummaryWriter):
        self._logger = logger

    def run(self, model: nn.Module, dataset: VisionDataset, config: TrainerConfig):

        counters = Counter()

        gen_optimizer, disc_optimizer = model.configure_optimizers(
            learning_rate=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # TODO manage resume
        model.initialize_weights()

        model.generator, gen_optimizer = self.setup(
            model.generator,
            gen_optimizer,
        )

        model.discriminator, disc_optimizer = self.setup(
            model.discriminator,
            disc_optimizer,
        )

        # Run All Configurations
        for step_config in config.get_step_configs(
                model.num_progression,
                len(dataset),
            ):

            img_size = model.get_image_size(progression_step=step_config["step"])

            dataset.transform = model.get_transform(img_size)

            dataloader = DataLoader(
                dataset,
                batch_size=step_config["batch_size"],
                num_workers=config.num_workers,
                shuffle=True,
            )
            dataloader = self.setup_dataloaders(dataloader)

            self.logger.add_scalar(
                "Progression/step",
                step_config["step"],
                counters["configuration_step"],
            )
            self.logger.add_scalar(
                "Progression/image_size",
                img_size,
                counters["configuration_step"],
            )
            self.logger.add_scalar(
                "Progression/batch_size",
                step_config["batch_size"],
                counters["configuration_step"],
            )

            # Run N Epochs
            for _ in tqdm(range(step_config["epochs"]), desc="running step {}".format(step_config["step"])):
                gen_losses = list()
                disc_losses = list()

                # Run Whole Dataset
                for iter_idx, (x_real, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                    batch_size = x_real.size(0)

                    # discriminator optimization
                    noise = generate_noise(model.latent_dim, batch_size=batch_size).to(self.device)
                    disc_optimizer.zero_grad()
                    disc_loss = model.compute_discriminator_loss(
                        x_real,
                        noise,
                        progression_step=step_config["step"],
                        alpha=step_config["alpha"],
                        gp_lambda=config.gp_lambda,
                        eps_drift=config.eps_drift,
                    )
                    self.backward(disc_loss, model=model.discriminator)
                    disc_optimizer.step()

                    # generator optimization
                    noise = generate_noise(model.latent_dim, batch_size=batch_size).to(self.device)
                    gen_optimizer.zero_grad()
                    gen_loss = model.compute_generator_loss(
                        noise,
                        progression_step=step_config["step"],
                        alpha=step_config["alpha"],
                    )
                    self.backward(gen_loss, model=model.generator)
                    gen_optimizer.step()

                    disc_losses.append(disc_loss.item())
                    gen_losses.append(gen_loss.item())

                    # update alpha inplace
                    step_config["alpha"] += step_config["d_alpha"]
                    step_config["alpha"] = min(step_config["alpha"], 1)

                    if (iter_idx + 1) % config.log_every_n_iter == 0:
                        self.logger.add_scalars(
                            "Loss",
                            dict(
                                generator=sum(gen_losses) / len(gen_losses),
                                discriminator=sum(disc_losses) / len(disc_losses),
                            ),
                            counters["iter_step"],
                        )
                        self.logger.add_scalar(
                            "Progression/alpha",
                            step_config["alpha"],
                            counters["iter_step"],
                        )

                        counters["iter_step"] += 1
                        gen_losses = list()
                        disc_losses = list()

            noise = generate_noise(model.latent_dim, batch_size=8).to(self.device)
            with torch.no_grad():
                fake_images = model.generator(
                    noise,
                    progression_step=step_config["step"],
                    alpha=1.0,
                )
            self.logger.add_images(
                "generated images",
                ((fake_images.cpu().numpy() + 1) / 2).clip(min=0, max=1),
                global_step=counters["configuration_step"],
            )

            # TODO save model

            counters["configuration_step"] += 1