from GenericPipeline.ArgumentParser import *
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.backends import cudnn
from torch import GradScaler
from tqdm import tqdm

import time

class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def test_nn(config):

    device_name = get_device(config)
    device = torch.device(device_name)
    cudnn.benchmark = True
    pin_memory = True
    enable_half = True
    if device_name == "cpu":
        enable_half = False  # Disable for CPU, it is slower!
    scaler = GradScaler(device_name, enabled=enable_half)

    train_set, test_set = get_dataset(config, get_transforms(config))
    train_set = SimpleCachedDataset(train_set)
    test_set = SimpleCachedDataset(test_set)

    train_loader, test_loader = get_loaders(config, train_set, test_set)

    model = get_model(config).to(device)
    model = torch.jit.script(model)
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(config, model)

    scheduler = get_scheduler(config, optimizer)

    early_stopping = get_early_stopping(config)

    def train():
        model.train()
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def val():
        model.eval()
        correct = 0
        total = 0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total, loss

    @torch.inference_mode()
    def inference():
        model.eval()

        labels = []

        for inputs, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)

            predicted = outputs.argmax(1).tolist()
            labels.extend(predicted)

        return labels

    best = 0.0
    epochs = list(range(25))

    # Just in order to make my prints not mess with each other
    time.sleep(5)

    with tqdm(epochs) as tbar:
        for epoch in tbar:
            train_acc = train()

            val_acc, val_loss = val()
            if val_acc > best:
                best = val_acc
            tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")

            # Early stopping mechanism
            if early_stopping[0] is True:
                if val_loss < early_stopping[2]:
                    early_stopping[2] = val_loss # Best loss
                    early_stopping[3] = early_stopping[1] # Reset to initial value
                else:
                    early_stopping[3] -= 1 # Decreasing counter
                    if early_stopping[3] == 0:
                        break

            # Scheduler mechanism
            if scheduler is not None:
                scheduler.step()

# Start Generic Pipeline

test_settings = {
    "runtime_method": SETTINGS_RUNTIME_METHOD_GPU,
    "dataset": SETTINGS_DATASET_CIFAR100,
    "scheduler": SETTINGS_SCHEDULER_NONE,

    "early_stopping": SETTINGS_EARLY_STOPPING_ON,

    "learning_rate": 0.05
}

models = [SETTINGS_MODEL_PRE_ACT_RESNET, SETTINGS_MODEL_VGG16]
optimizers = [SETTINGS_OPTIMIZER_SGD_NESTEROV, SETTINGS_OPTIMIZER_ADAM]
transforms = [[SETTINGS_TRANSFORMS_HORIZONTAL_FLIP, SETTINGS_TRANSFORMS_VERTICAL_FLIP, SETTINGS_TRANSFORMS_NORMALIZE],
              [SETTINGS_TRANSFORMS_RANDOM_CROP, SETTINGS_TRANSFORMS_GRAYSCALE, SETTINGS_TRANSFORMS_NORMALIZE]]

for model in models:
    for optimizer in optimizers:
        for transform in transforms:
            test_settings['model'] = model
            test_settings['optimizer'] = optimizer
            test_settings['transform'] = transform

            print("========================")

            print(f"Using model {model}, optimizer {optimizer}, transform {transform}")

            test_nn(test_settings)

            print("========================")