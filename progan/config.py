from typing import Tuple, List, Dict

from pydantic import BaseModel


class TrainerConfig(BaseModel):
    max_epochs: int = 30
    initial_batch_size: int = 2**5
    batch_size_factors: List = None

    learning_rate: float = 1e-3
    weight_decay: float = 0
    betas: Tuple[float, float] = (0, 0.99)
    eps: float = 1e-8

    gp_lambda: float = 10
    eps_drift: float = 1e-3

    num_workers: int = 0

    log_every_n_iter: int = 20

    def get_step_configs(
        self,
        num_progression: int,
        num_samples: int,
    ) -> Dict:

        factors = self.batch_size_factors or [1] * num_progression

        assert len(factors) == num_progression

        step_configs = list()
        step_epochs = self.max_epochs // ((num_progression * 2) + 1)

        step_configs.append({
            "epochs": step_epochs,
            "batch_size": self.initial_batch_size,
            "step": 0,
            "fade_in": False,
        })
        for i in range(num_progression):
            for j in range(2):
                step_configs.append({
                    "epochs": step_epochs,
                    "batch_size": int(self.initial_batch_size * factors[i]),
                    "step": i + 1,
                    "fade_in": j % 2 == 0,
                })

        # add leftover epochs
        step_configs[-1]["epochs"] += (self.max_epochs % ((num_progression * 2) + 1))

        # compute alpha & delta alpha
        for step_config in step_configs:
            if step_config["fade_in"]:
                iter_size = num_samples // step_config["batch_size"]
                step_config["alpha"] = 0
                step_config["d_alpha"] = 1/(step_config["epochs"] * iter_size)
            else:
                step_config["alpha"] = 1
                step_config["d_alpha"] = 0

        return step_configs


class ModelConfig(BaseModel):
    latent_dim: int = 2**9
    img_channels: int = 3
    final_img_size: int = 1024

    # add ons
    use_wscale: bool = True
    use_pixelnorm: bool = True
    use_mb_stddev: bool = True