import os
from typing import Optional

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = "./data"
DEFAULT_NUM_WORKERS = 32

TRANSFORM_IMAGENET = [
    transforms.Resize(40),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
]


def train_val_split(dataset_size: int, val_train_split: float = 0.02):
    validation_split = int((1 - val_train_split) * dataset_size)
    train_indices = range(dataset_size)
    train_indices, val_indices = (
        train_indices[:validation_split],
        train_indices[validation_split:],
    )
    return train_indices, val_indices


def extract_attack_set(
    dataset: Dataset,
    sample_per_class: int = 5,
    multi_class=False,
    total_num_samples: int = 50,
    seed: int = None,
):
    if not multi_class:
        num_classes = len(dataset.classes)
        class2sample = {i: [] for i in range(num_classes)}
        select_indices = []
        if seed == None:
            index_pool = range(len(dataset))
        else:
            index_pool = np.random.RandomState(seed=seed).permutation(
                len(dataset))
        for i in index_pool:
            current_class = dataset[i][1]
            if len(class2sample[current_class]) < sample_per_class:
                class2sample[current_class].append(i)
                select_indices.append(i)
            elif len(select_indices) == sample_per_class * num_classes:
                break
        return select_indices, class2sample
    else:
        select_indices = range(total_num_samples)
        class2sample = None
        return select_indices, class2sample


class FileDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        transform: torch.nn.Module = transforms.Compose(TRANSFORM_IMAGENET),
        batch_size: int = 32,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform
        self.batch_sampler = batch_sampler

    def setup(self, stage: Optional[str] = None):
        self.dataset = datasets.ImageFolder(self.data_dir,
                                            transform=self.transform)

    def get_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return self.get_dataloader()

    def test_dataloader(self):
        return self.get_dataloader()


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        augment: dict = None,
        data_dir: str = os.path.join(DEFAULT_DATA_DIR, "imagenet"),
        batch_size: int = 32,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        tune_on_val: bool = False,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = 1000
        self.multi_class = False

        self.batch_sampler = batch_sampler
        self.tune_on_val = tune_on_val

        print(data_dir)
        imagenet_normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                                  (0.229, 0.224, 0.225))

        self._train_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
        if augment["hflip"]:
            self._train_transforms.insert(
                0, transforms.RandomHorizontalFlip(p=0.5))
        if augment["color_jitter"] is not None:
            self._train_transforms.insert(
                0,
                transforms.ColorJitter(
                    brightness=augment["color_jitter"][0],
                    contrast=augment["color_jitter"][1],
                    saturation=augment["color_jitter"][2],
                    hue=augment["color_jitter"][3],
                ),
            )
        if augment["rotation"] > 0:
            self._train_transforms.insert(
                0, transforms.RandomRotation(augment["rotation"]))
        if augment["crop"]:
            self._train_transforms.insert(0,
                                          transforms.RandomCrop(32, padding=4))

        print(self._train_transforms)

        self._test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet_normalize,
        ]

    def setup(self, stage: Optional[str] = None):
        """Initialize the dataset based on the stage option ('fit', 'test' or 'attack'):
        - if stage is 'fit', set up the training and validation dataset;
        - if stage is 'test', set up the testing dataset;
        - if stage is 'attack', set up the attack dataset (a subset of training images)

        Args:
            stage (Optional[str], optional): stage option. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_set = datasets.ImageFolder(
                os.path.join(self.data_dir, "train"),
                transform=transforms.Compose(self._train_transforms),
            )
            if self.tune_on_val:
                self.val_set = datasets.ImageFolder(
                    os.path.join(self.data_dir, "train"),
                    transform=transforms.Compose(self._test_transforms),
                )
                train_indices, val_indices = train_val_split(
                    len(self.train_set), self.tune_on_val)
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)
            else:  # use test set
                self.val_set = datasets.ImageFolder(
                    os.path.join(self.data_dir, "val"),
                    transform=transforms.Compose(self._test_transforms),
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = datasets.ImageFolder(
                os.path.join(self.data_dir, "val"),
                transform=transforms.Compose(self._test_transforms),
            )

        if stage == "attack":
            ori_train_set = datasets.ImageFolder(
                os.path.join(self.data_dir, "attack"),
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set)
            self.train_set = Subset(ori_train_set, self.attack_indices)

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(self.train_set,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        augment: dict = None,
        batch_size: int = 32,
        data_dir: str = DEFAULT_DATA_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        tune_on_val: float = 0,
    ):
        super().__init__()
        self._has_setup_attack = False

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10

        self.batch_sampler = batch_sampler
        self.tune_on_val = tune_on_val
        self.multi_class = False

        mnist_normalize = transforms.Normalize((0.1307, ), (0.3081, ))

        self._train_transforms = [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            mnist_normalize,
        ]
        if augment["hflip"]:
            self._train_transforms.insert(
                0, transforms.RandomHorizontalFlip(p=0.5))
        if augment["color_jitter"] is not None:
            self._train_transforms.insert(
                0,
                transforms.ColorJitter(
                    brightness=augment["color_jitter"][0],
                    contrast=augment["color_jitter"][1],
                    saturation=augment["color_jitter"][2],
                    hue=augment["color_jitter"][3],
                ),
            )
        if augment["rotation"] > 0:
            self._train_transforms.insert(
                0, transforms.RandomRotation(augment["rotation"]))
        if augment["crop"]:
            self._train_transforms.insert(0,
                                          transforms.RandomCrop(32, padding=4))

        print(self._train_transforms)

        self._test_transforms = [
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            mnist_normalize,
        ]

        self.prepare_data()

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Initialize the dataset based on the stage option ('fit', 'test' or 'attack'):
        - if stage is 'fit', set up the training and validation dataset;
        - if stage is 'test', set up the testing dataset;
        - if stage is 'attack', set up the attack dataset (a subset of training images)

        Args:
            stage (Optional[str], optional): stage option. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_set = MNIST(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            if self.tune_on_val:
                self.val_set = MNIST(
                    self.data_dir,
                    train=True,
                    transform=transforms.Compose(self._test_transforms),
                )
                train_indices, val_indices = train_val_split(
                    len(self.train_set), self.tune_on_val)
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)
            else:
                self.val_set = MNIST(
                    self.data_dir,
                    train=False,
                    transform=transforms.Compose(self._test_transforms),
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = MNIST(
                self.data_dir,
                train=False,
                transform=transforms.Compose(self._test_transforms),
            )

        if stage == "attack":
            ori_train_set = MNIST(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_mini":
            ori_train_set = MNIST(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=2)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_large":
            ori_train_set = MNIST(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=500)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        augment: dict = None,
        batch_size: int = 32,
        data_dir: str = DEFAULT_DATA_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        batch_sampler: Sampler = None,
        tune_on_val: float = 0,
        seed: int = None,
    ):
        super().__init__()
        self._has_setup_attack = False

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.num_classes = 10

        self.batch_sampler = batch_sampler
        self.tune_on_val = tune_on_val
        self.multi_class = False
        self.seed = seed

        cifar_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                               (0.2023, 0.1994, 0.2010))

        self._train_transforms = [transforms.ToTensor(), cifar_normalize]
        if augment["hflip"]:
            self._train_transforms.insert(
                0, transforms.RandomHorizontalFlip(p=0.5))
        if augment["color_jitter"] is not None:
            self._train_transforms.insert(
                0,
                transforms.ColorJitter(
                    brightness=augment["color_jitter"][0],
                    contrast=augment["color_jitter"][1],
                    saturation=augment["color_jitter"][2],
                    hue=augment["color_jitter"][3],
                ),
            )
        if augment["rotation"] > 0:
            self._train_transforms.insert(
                0, transforms.RandomRotation(augment["rotation"]))
        if augment["crop"]:
            self._train_transforms.insert(0,
                                          transforms.RandomCrop(32, padding=4))

        print(self._train_transforms)

        self._test_transforms = [transforms.ToTensor(), cifar_normalize]

        self.prepare_data()

    def prepare_data(self):
        """Download the data"""
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Initialize the dataset based on the stage option ('fit', 'test' or 'attack'):
        - if stage is 'fit', set up the training and validation dataset;
        - if stage is 'test', set up the testing dataset;
        - if stage is 'attack', set up the attack dataset (a subset of training images)

        Args:
            stage (Optional[str], optional): stage option. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            if self.tune_on_val:
                self.val_set = CIFAR10(
                    self.data_dir,
                    train=True,
                    transform=transforms.Compose(self._test_transforms),
                )
                train_indices, val_indices = train_val_split(
                    len(self.train_set), self.tune_on_val)
                self.train_set = Subset(self.train_set, train_indices)
                self.val_set = Subset(self.val_set, val_indices)
            else:
                self.val_set = CIFAR10(
                    self.data_dir,
                    train=False,
                    transform=transforms.Compose(self._test_transforms),
                )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = CIFAR10(
                self.data_dir,
                train=False,
                transform=transforms.Compose(self._test_transforms),
            )

        if stage == "attack":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, seed=self.seed)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_mini":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=2)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))
        elif stage == "attack_large":
            ori_train_set = CIFAR10(
                self.data_dir,
                train=True,
                transform=transforms.Compose(self._train_transforms),
            )
            self.attack_indices, self.class2attacksample = extract_attack_set(
                ori_train_set, sample_per_class=500)
            self.train_set = Subset(ori_train_set, self.attack_indices)
            self.test_set = Subset(self.test_set, range(100))

    def train_dataloader(self):
        if self.batch_sampler is None:
            return DataLoader(self.train_set,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)
        else:
            return DataLoader(
                self.train_set,
                batch_sampler=self.batch_sampler,
                num_workers=self.num_workers,
            )

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
