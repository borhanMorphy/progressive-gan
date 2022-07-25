import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as F

def compute_gradient_penalty(x_real: Tensor, x_fake: Tensor, disc: nn.Module, progression_step: float = 0) -> Tensor:
    batch_size = x_real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=x_real.device, dtype=x_real.dtype)
    x_hat = alpha*x_real + (1-alpha) * x_fake
    x_hat.requires_grad = True

    scores = disc(x_hat, progression_step=progression_step)
    # compute gradients with respect to `x_hat`
    gradient, = torch.autograd.grad(
        inputs=x_hat,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )
    # gradient: batch_size, C, H, W

    # compute l2 norm of gradients 
    norm = gradient.flatten(start_dim=1).norm(p=2, dim=1)
    # norm: batch_size,

    return (norm - 1) ** 2

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_s = max(w, h)
		left = (max_s - w) // 2
		right = left + (max_s - w) % 2
		top = (max_s - h) // 2
		bottom = top + (max_s - h) % 2

		return F.pad(image, (left, top, right, bottom), 0, 'constant')

def generate_noise(latent_dim: float, batch_size: int = 1) -> Tensor:
    # Gaussian(mean=0, std=1)
    return torch.randn(batch_size, latent_dim, 1, 1)
