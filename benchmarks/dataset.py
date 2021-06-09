import numpy as np
from numpy.random import sample
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torch.utils.data import DataLoader


class CIFAR10Data:
    def __init__(
        self,
        transform_train,
        transform_test,
        shuffle: bool = False,
        num_workers: int = 0,
        create_val: bool = True,
        val_size: float = 0.2,
        batch_size: int = 128,
        random_seed: int = 1,
    ) -> None:
        self.shuffle = shuffle
        self.val_size = val_size
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.seed = random_seed
        self.create_val = create_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _download_data(self):
        train_data = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform_train
        )
        test_data = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform_test
        )

        return train_data, test_data

    def _create_val(self, train_data):
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(self.val_size * num_train)

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader

    def initialize_dataset(self):
        train_data, test_data = self._download_data()

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if self.create_val:
            train_loader, val_loader = self._create_val(train_data)
            return train_loader, val_loader, test_loader
        else:
            return train_loader, test_loader


class ImageNetData:
    def __init__(
        self,
        transform_train,
        transform_test,
        shuffle: bool = False,
        num_workers: int = 0,
        create_val: bool = True,
        val_size: float = 0.2,
        batch_size: int = 128,
        random_seed: int = 1,
    ) -> None:
        self.shuffle = shuffle
        self.val_size = val_size
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.seed = random_seed
        self.create_val = create_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _download_data(self):
        train_data = datasets.ImageNet(
            root="./data", train=True, download=True, transform=self.transform_train
        )
        test_data = datasets.ImageNet(
            root="./data", train=False, download=True, transform=self.transform_test
        )

        return train_data, test_data

    def _create_val(self, train_data):
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(self.val_size * num_train)

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

        val_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader

    def initialize_dataset(self):
        train_data, test_data = self._download_data()

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if self.create_val:
            train_loader, val_loader = self._create_val(train_data)
            return train_loader, val_loader, test_loader
        else:
            return train_loader, test_loader
