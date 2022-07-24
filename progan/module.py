import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import Generator
from .discriminator import Discriminator

class ProGAN(nn.Module):
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self):
        # TODO
        pass

    def compute_loss(self):
        pass