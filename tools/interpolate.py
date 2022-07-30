import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
from moviepy.editor import ImageSequenceClip

import progan


def main(args):
    model = progan.ProGAN.from_checkpoint(
        args.ckpt_path
    )

    sample_noises = progan.utils.generate_noise(
        model.latent_dim,
        batch_size=args.num_sample_points,
    ).numpy()

    ids = [(i % args.num_sample_points, (i+1) % args.num_sample_points) for i in range(args.num_sample_points)]

    frames = list()
    for source_idx, target_idx in tqdm(ids):
        batch = np.linspace(sample_noises[source_idx], sample_noises[target_idx], args.fps)
        batch = torch.from_numpy(batch.reshape(args.fps, -1, 1, 1))
        for fake_img in (model(batch).cpu().permute(0, 2, 3, 1) * 255):
            frames.append(
                fake_img.numpy().astype(np.uint8)
            )

    os.makedirs(
        os.path.dirname(args.target_file),
        exist_ok=True
    )

    clip = ImageSequenceClip(frames, fps=args.fps)
    clip.write_gif(args.target_file, fps=args.fps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", "-ckpt", type=str)
    ap.add_argument("--num-sample-points", "-n", type=int, default=2)
    ap.add_argument("--target-file", "-t", type=str, default="resources/test.gif")
    ap.add_argument("--fps", "-f", type=int, default=10)

    args = ap.parse_args()

    main(args)