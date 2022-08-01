# Progressive Growing of GANs for Improved Quality, Stability, and Variation

**&mdash; Unofficial PyTorch implementation of the ICLR 2018 paper**


**Below images were generated by this repository**<br>
<img src="/resources/mnist.gif" width="128" height="128"/>

<!-- TODO add badges -->

<!-- TODO add interpoation visuals in here -->

## Contents
- [Installation](#installation)
- [Pretrained Models](#pretrained-models)
- [Demo](#demo)
- [Benchmarks](#benchmarks)
- [Tutorials](#tutorials)
- [References](#references)
- [Citations](#citations)

## Installation
```
git clone https://github.com/borhanMorphy/progressive-gan.git
cd progressive-gan
pip install .
```

## Pretrained Models
Dataset|Configuration|Resolution|Link
:------:|:------:|:------:|:------:
**MNIST**|[mnist](configs/mnist.yml)|32 x 32|-|


## References
- [Official Implementation](https://github.com/tkarras/progressive_growing_of_gans)
- [Paper](https://arxiv.org/pdf/1710.10196.pdf)

## Citations
```bibtex
@article{karras2017progressive,
    title={Progressive growing of gans for improved quality, stability, and variation},
    author={Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
    journal={arXiv preprint arXiv:1710.10196},
    year={2017},
    url={https://arxiv.org/abs/1710.10196},
}
```