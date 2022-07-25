from torchvision.datasets import MNIST, CelebA
import progan

def main():
    dataset = MNIST(
        root="./data",
        download=True,
        train=False,
    )

    model_config = progan.ModelConfig(
        latent_dim=2**7,
        img_channels=1,
        final_img_size=32,
    )

    train_config = progan.TrainerConfig()

    model = progan.ProGAN.from_config(model_config)

    trainer = progan.LiteTrainer(
        precision = 32,
        gpus=1,
    )

    trainer.run(
        model,
        dataset,
        train_config,
    )


if __name__ == '__main__':
    main()