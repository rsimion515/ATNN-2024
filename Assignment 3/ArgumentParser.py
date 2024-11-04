import json
import os.path

from torchvision.transforms import v2
import torch
from torchvision.datasets import CIFAR100, MNIST, CIFAR10
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import timm

from Models import *

SETTINGS_RUNTIME_METHOD_CPU = 1
SETTINGS_RUNTIME_METHOD_GPU = 2

def get_device(config):
    if config.get('runtime_method', SETTINGS_RUNTIME_METHOD_CPU) == SETTINGS_RUNTIME_METHOD_GPU:
        return "cuda"
    else:
        return "cpu"

SETTINGS_TRANSFORMS_HORIZONTAL_FLIP = 1
SETTINGS_TRANSFORMS_VERTICAL_FLIP = 2
SETTINGS_TRANSFORMS_RANDOM_CROP = 3
SETTINGS_TRANSFORMS_GRAYSCALE = 4
SETTINGS_TRANSFORMS_NORMALIZE = 5

def get_transforms(config):
    default_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    list_transforms = config.get("transforms", [])
    for transform in list_transforms:
        if transform == SETTINGS_TRANSFORMS_HORIZONTAL_FLIP:
            default_list.append(v2.RandomHorizontalFlip(0.5))
        elif transform == SETTINGS_TRANSFORMS_VERTICAL_FLIP:
            default_list.append(v2.RandomVerticalFlip(0.5))
        elif transform == SETTINGS_TRANSFORMS_RANDOM_CROP:
            resize_crop = config.get("resize_crop_size", 32)
            default_list.append(v2.RandomResizedCrop(size=resize_crop))
        elif transform == SETTINGS_TRANSFORMS_GRAYSCALE:
            default_list.append(v2.RandomGrayscale())
        elif transform == SETTINGS_TRANSFORMS_NORMALIZE:
            normalize_mean = config.get("normalize_mean", (0.485, 0.456, 0.406))
            normalize_std = config.get("normalize_std", (0.229, 0.224, 0.225))
            default_list.append(v2.Normalize(normalize_mean, normalize_std))
    return v2.Compose(default_list)

SETTINGS_DATASET_MNIST    = 1
SETTINGS_DATASET_CIFAR10  = 2
SETTINGS_DATASET_CIFAR100 = 3

def get_dataset(config, transforms):
    selected_dataset = config.get('dataset', SETTINGS_DATASET_MNIST)

    dataset_location = os.path.join(os.path.curdir, "datasets")
    print("Saving datasets to", dataset_location)
    if os.path.exists(dataset_location) is False:
        print("Creating datasets folder")
        os.makedirs(dataset_location)

    if selected_dataset == SETTINGS_DATASET_MNIST:
        train_set = MNIST(dataset_location, download=True, train=True, transform=transforms)
        test_set = MNIST(dataset_location, download=True, train=False, transform=transforms)
    elif selected_dataset == SETTINGS_DATASET_CIFAR10:
        train_set = CIFAR10(dataset_location, download=True, train=True, transform=transforms)
        test_set = CIFAR10(dataset_location, download=True, train=False, transform=transforms)
    elif selected_dataset == SETTINGS_DATASET_CIFAR100:
        train_set = CIFAR100(dataset_location, download=True, train=True, transform=transforms)
        test_set = CIFAR100(dataset_location, download=True, train=False, transform=transforms)
    else:
        train_set = MNIST(dataset_location, download=True, train=True, transform=transforms)
        test_set = MNIST(dataset_location, download=True, train=False, transform=transforms)

    return train_set, test_set

def get_loaders(config, train_set, test_set):
    batch_size_train = config.get('batch_size_train', 64)
    batch_size_test = config.get('batch_size_test', 500)
    shuffle = config.get('shuffle', True)
    pin_memory = config.get('pin_memory', True)

    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=shuffle, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, pin_memory=pin_memory)

    return train_loader, test_loader

SETTINGS_MODEL_MLP              = 1
SETTINGS_MODEL_LE_NET           = 2
SETTINGS_MODEL_RESNET18         = 3
SETTINGS_MODEL_PRE_ACT_RESNET   = 4
SETTINGS_MODEL_VGG16            = 5

def get_model(config):
    selected_model = config.get('model', SETTINGS_MODEL_MLP)
    selected_dataset = config.get('dataset', SETTINGS_DATASET_MNIST)

    if selected_model == SETTINGS_MODEL_MLP:
        print("Using MLP")
        return MLP()
    elif selected_model == SETTINGS_MODEL_LE_NET:
        print("Using LeNet")
        return LeNet()
    elif selected_model == SETTINGS_MODEL_VGG16:
        print("Using VGG16")
        return VGG16()
    elif selected_model == SETTINGS_MODEL_PRE_ACT_RESNET:
        print("Using PreActResNet")
        if selected_dataset == SETTINGS_DATASET_CIFAR10:
            return PreActResNet18(10)
        elif selected_dataset == SETTINGS_DATASET_CIFAR100:
            return PreActResNet18(100)
        else:
            return None
    elif selected_model == SETTINGS_MODEL_RESNET18:
        print("Using ResNet18 from timm")
        if selected_dataset == SETTINGS_DATASET_CIFAR10:
            return timm.create_model('resnet18_cifar10', pretrained=True)
        else:
            return None
    else:
        print("Using None")
        return None

SETTINGS_OPTIMIZER_SGD              = 1
SETTINGS_OPTIMIZER_SGD_MOMENTUM     = 2
SETTINGS_OPTIMIZER_SGD_NESTEROV     = 3
SETTINGS_OPTIMIZER_SGD_WEIGHT_DECAY = 4
SETTINGS_OPTIMIZER_ADAM             = 5
SETTINGS_OPTIMIZER_ADAM_W           = 6
SETTINGS_OPTIMIZER_RMS_PROP         = 7

def get_optimizer(config, model):
    selected_optimizer = config.get('optimizer', SETTINGS_OPTIMIZER_SGD)

    learning_rate = config.get('learning_rate', 0.01)

    if selected_optimizer == SETTINGS_OPTIMIZER_SGD:
        print("Using SETTINGS_OPTIMIZER_SGD")
        current_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif selected_optimizer == SETTINGS_OPTIMIZER_SGD_MOMENTUM:
        print("Using SETTINGS_OPTIMIZER_SGD_MOMENTUM")
        current_momentum = config.get('momentum', 0.9)
        current_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=current_momentum)
    elif selected_optimizer == SETTINGS_OPTIMIZER_SGD_NESTEROV:
        print("Using SETTINGS_OPTIMIZER_SGD_NESTEROV")
        current_momentum = config.get('momentum', 0.9)
        current_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=current_momentum, nesterov=True)
    elif selected_optimizer == SETTINGS_OPTIMIZER_SGD_WEIGHT_DECAY:
        print("Using SETTINGS_OPTIMIZER_SGD_WEIGHT_DECAY")
        decay = config.get('weight_decay', 0)
        current_optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif selected_optimizer == SETTINGS_OPTIMIZER_ADAM:
        print("Using SETTINGS_OPTIMIZER_ADAM")
        current_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif selected_optimizer == SETTINGS_OPTIMIZER_ADAM_W:
        print("Using SETTINGS_OPTIMIZER_ADAM_W")
        current_optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif selected_optimizer == SETTINGS_OPTIMIZER_RMS_PROP:
        print("Using SETTINGS_OPTIMIZER_RMS_PROP")
        current_optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        print("Using SETTINGS_OPTIMIZER_SGD")
        current_optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return current_optimizer

SETTINGS_SCHEDULER_NONE                 = 1
SETTINGS_SCHEDULER_STEP_LR              = 2
SETTINGS_SCHEDULER_REDUCE_LR_ON_PLATEAU = 3

def get_scheduler(config, optimizer):
    selected_scheduler = config.get('scheduler', SETTINGS_SCHEDULER_NONE)

    if selected_scheduler == SETTINGS_SCHEDULER_STEP_LR:
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        current_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif selected_scheduler == SETTINGS_SCHEDULER_REDUCE_LR_ON_PLATEAU:
        current_scheduler = ReduceLROnPlateau(optimizer)
    else:
        current_scheduler = None

    return current_scheduler

SETTINGS_EARLY_STOPPING_OFF = 1
SETTINGS_EARLY_STOPPING_ON  = 2

def get_early_stopping(config):
    if config.get('early_stopping', SETTINGS_EARLY_STOPPING_OFF) == SETTINGS_EARLY_STOPPING_ON:
        early_stopping_patience = config.get('early_stopping_patience', 10)
        return [True, early_stopping_patience, float("inf"), early_stopping_patience]
    else:
        return [False]

default_settings = {
    "runtime_method": SETTINGS_RUNTIME_METHOD_GPU,
    "dataset": SETTINGS_DATASET_CIFAR100,
    "model": SETTINGS_MODEL_PRE_ACT_RESNET,
    "optimizer": SETTINGS_OPTIMIZER_SGD,
    "scheduler": SETTINGS_SCHEDULER_NONE,
    "early_stopping": SETTINGS_EARLY_STOPPING_OFF,
    "transforms": [SETTINGS_TRANSFORMS_NORMALIZE]
}

def get_config():
    if os.path.exists("config.json"):
        with open("config.json") as file:
            config = json.load(file)
    else:
        config = default_settings

    return config