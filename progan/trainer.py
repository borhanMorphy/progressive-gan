from typing import Tuple, Dict
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning.lite import LightningLite

from torchvision.datasets.vision import VisionDataset

from .config import TrainerConfig
from .utils import generate_noise

class LiteTrainer(LightningLite):
    def run(self, model: nn.Module, dataset: VisionDataset, config: TrainerConfig):

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

        for step_config in config.get_step_configs(
                model.num_progression,
                len(dataset),
            ):

            dataset.transform = model.get_transform(
                model.get_image_size(progression_step=step_config["step"])
            )

            dataloader = DataLoader(
                dataset,
                batch_size=step_config["batch_size"],
                num_workers=config.num_workers,
                shuffle=True,
            )
            dataloader = self.setup_dataloaders(dataloader)

            pbar = tqdm(range(step_config["epochs"]))
            for _ in pbar:
                # TODO log losses
                # run single epoch
                gen_losses, dis_losses = self.run_single_epoch(
                    model,
                    dataloader,
                    (gen_optimizer, disc_optimizer),
                    config,
                    step_config,
                    pbar, # TODO handle pbar
                )

    def run_single_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizers: Tuple,
        config: TrainerConfig,
        step_config: Dict,
        pbar,
    ):
        description_template = "step: {}\talpha: {:.03f}"
        # TODO use logger instead
        gen_losses = list()
        disc_losses = list()
        gen_optim, disc_optim = optimizers
        for x_real, _ in tqdm(dataloader):
            batch_size = x_real.size(0)

            noise = generate_noise(model.latent_dim, batch_size=batch_size).to(self.device)
            disc_optim.zero_grad()
            disc_loss = model.compute_discriminator_loss(
                x_real,
                noise,
                progression_step=step_config["step"],
                alpha=step_config["alpha"],
                gp_lambda=config.gp_lambda,
                eps_drift=config.eps_drift,
            )
            self.backward(disc_loss, model=model.discriminator)
            disc_optim.step()

            noise = generate_noise(model.latent_dim, batch_size=batch_size).to(self.device)
            gen_optim.zero_grad()
            gen_loss = model.compute_generator_loss(
                noise,
                progression_step=step_config["step"],
                alpha=step_config["alpha"],
            )
            self.backward(gen_loss, model=model.generator)
            gen_optim.step()

            disc_losses.append(disc_loss.item())
            gen_losses.append(gen_loss.item())

            # update alpha inplace
            step_config["alpha"] += step_config["d_alpha"]
            step_config["alpha"] = min(step_config["alpha"], 1)

            pbar.set_description(description_template.format(step_config["step"], step_config["alpha"]))

        return gen_losses, disc_losses
