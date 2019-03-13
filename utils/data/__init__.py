# -*- coding: utf-8 -*-

from torchvision import datasets, transforms

from .cifar_subset import *
from .imagenet import *
from .sampler import *

__all__ = ["get_dataset", "IterationSampler"]

_dataset_zoo = {
    "CIFAR100Sub": CIFAR100Sub,
    "ImageNet": ImageNetDataset,
}
_dataset_zoo.update({k: v for k, v in vars(datasets).items()
                     if not k.startswith("__")})

_normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
_train_transforms = {
    "CIFAR10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize_rgb,
    ]),
    "ImageNet": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize_rgb,
    ]),
}
_test_transforms = {
    "CIFAR10": transforms.Compose([
        transforms.ToTensor(),
        _normalize_rgb,
    ]),
    "ImageNet": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _normalize_rgb,
    ]),
}


def get_dataset(name, *args, **kwargs):
    train_trans = _train_transforms[name]
    test_trans = _test_transforms[name]
    train_set = _dataset_zoo[name](*args, train=True, transform=train_trans, **kwargs)
    test_set = _dataset_zoo[name](*args, train=False, transform=test_trans, **kwargs)

    return train_set, test_set
