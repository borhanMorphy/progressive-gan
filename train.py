import argparse
from typing import Tuple

import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST, CelebA, CIFAR10, VisionDataset
import progan

def load_config(yml_file_path: str) -> Tuple[progan.ModelConfig, progan.TrainerConfig]:

    with open(yml_file_path, "r") as foo:
        config = yaml.load(foo, yaml.FullLoader)

    model_config = progan.ModelConfig(
        **config.get("model_config", dict())
    )

    trainer_config = progan.TrainerConfig(
        **config.get("trainer_config", dict())
    )

    return model_config, trainer_config

def get_dataset(dataset_name: str) -> VisionDataset:
    if dataset_name.lower() == "mnist":
        return MNIST("data", train=True, download=True)
    elif dataset_name.lower() == "celeba":
        return CelebA("data", split="train", download=True)
    elif dataset_name.lower() == "cifar10":
        return CIFAR10("data", train=True, download=True)
    else:
        raise AssertionError("dataset {} not found".format(dataset_name))

def main(args):
    model_config, train_config = args.config

    model = progan.ProGAN.from_config(model_config)

    trainer = progan.LiteTrainer(
        precision=args.precision,
        gpus=args.gpus,
    )

    trainer.logger = SummaryWriter(log_dir=args.log_dir)
    trainer.checkpoint_path = args.checkpoint_path
    trainer.model_name = args.model_name

    if args.seed is not None:
        trainer.seed_everything(seed=args.seed)

    trainer.run(
        model,
        args.dataset,
        train_config,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=load_config, required=True)
    ap.add_argument("--dataset", "-d", type=get_dataset, required=True)
    ap.add_argument("--gpus", "-g", type=int, default=1)
    ap.add_argument("--precision", "-p", type=int, default=32, choices=[16, 32])
    ap.add_argument("--seed", "-s", type=int)
    ap.add_argument("--log-dir", "-l", type=str)
    ap.add_argument("--checkpoint-path", type=str)
    ap.add_argument("--model-name", "-m", type=str)

    args = ap.parse_args()
    main(args)