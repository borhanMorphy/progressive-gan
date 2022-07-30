import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST, CelebA
import progan

def main(args):

    dataset = MNIST(
        root="./data",
        download=True,
        train=True,
    )

    model_config = progan.ModelConfig(
        latent_dim=2**7,
        img_channels=1,
        final_img_size=32,
    )

    train_config = progan.TrainerConfig()

    model = progan.ProGAN.from_config(model_config)

    trainer = progan.LiteTrainer(
        precision=args.precision,
        gpus=args.gpus,
    )

    trainer.logger = SummaryWriter(log_dir=args.log_dir)
    trainer.checkpoint_path = args.checkpoint_path

    if args.seed is not None:
        trainer.seed_everything(seed=args.seed)

    trainer.run(
        model,
        dataset,
        train_config,
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", "-g", type=int, default=1)
    ap.add_argument("--precision", "-p", type=int, default=32, choices=[16, 32])
    ap.add_argument("--seed", "-s", type=int)
    ap.add_argument("--log-dir", "-l", type=str)
    ap.add_argument("--checkpoint-path", type=str)

    args = ap.parse_args()
    main(args)