# Overview of AIML I have learned

![lint](https://github.com/Beomus/py-dsa/actions/workflows/black.yml/badge.svg)
![unittest](https://github.com/Beomus/py-dsa/actions/workflows/unittest.yml/badge.svg)
[![CircleCI](https://circleci.com/gh/Beomus/AIML-Review.svg?style=shield)](https://github.com/Beomus/AIML-Review/)

Review notes can be found [here](https://docs.google.com/document/d/1ocJ-YzZ6IvvCjJWNqE98Q6tpFOXDguBGV7xBZhqHlss/edit?usp=sharing) (please feel free to leave any comments and suggestion and correction).

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
- [ ] NLP

Frameworks:

- [x] PyTorch
- [x] TensorFlow
- [ ] FastAI

## Usage

`conda env create -f environment.yml`

`conda activate aiml`

### *Note*

- TensorFlow installation is not included please check carefully on how to install TensorFlow that is compatible with your machine.
- In case environment does not work because of OS differences, trying installing dependencies with pip:

`conda create --name aiml python=3.8`

`conda activate aiml`

`pip install -r requirements.txt`

Please feel free to post any issues or PR as needed, I will try to review them as much as I can. Thanks!
