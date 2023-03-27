# Overview of Deep Learning I have learned

![lint](https://github.com/Beomus/py-dsa/actions/workflows/black.yml/badge.svg)
![unittest](https://github.com/Beomus/py-dsa/actions/workflows/unittest.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/Beomus/AIML-Review.svg?style=shield)](https://github.com/Beomus/AIML-Review/)

Review notes can be found [here](./docs/notes.md) (please feel free to leave any comments and suggestion and correction).

---
I will try to cover as much as I can. However, this repo is not meant to give you any advanced information or make you an expert at a particular subject.

This can be simply understood as my summarization of techniques and methods both theoretical and practical on Deep Learning, including PyTorch and TensorFlow.

---
> **I will finalized the repo structure onced I have gone over my notes**

## Additional info

> _Notes: items being crossed off means that they are included but might not completely cover the subject matter._

Concepts:

- [x] Deep Learning in general
- [x] Activation functions
- [x] Hyperparameters tuning
- [x] Image recognition
- [x] Image segmentation
- [x] NLP

Frameworks:

- [x] PyTorch
- [x] TensorFlow
- [ ] FastAI

## Installation

I purposely use Windows 11 🤮 because I'm too lazy to setup dual-boot on my
laptop. Therefore, this instructions will be geared towards W11.

As of March 20th, 2023, [PyTorch](https://pytorch.org/get-started/locally/) supports [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network) so we will start
from there.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Other libs:

- ml-collections
- sklearn
- tensorboard
- tqdm
- einops
- Pillow
