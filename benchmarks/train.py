from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
