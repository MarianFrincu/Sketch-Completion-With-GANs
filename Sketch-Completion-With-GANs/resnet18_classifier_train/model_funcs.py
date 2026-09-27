import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

from models.resnet18_classifier import IMAGENET_MEAN, IMAGENET_STD
from util.c_dataset import CDataset
from util.image_functions import random_shift


def prepare_data(batch_size, num_workers, train_size, paths):

    datasets = [ImageFolder(root=path) for path in paths]

    if any(dataset.classes != datasets[0].classes for dataset in datasets):
        raise ValueError("All data folders must contain the same classes")

    combined_dataset = ConcatDataset(datasets)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(random_shift),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    targets = [target for dataset in datasets for target in dataset.targets]

    train_idx, val_idx = train_test_split(
        range(len(combined_dataset)),
        train_size=train_size, shuffle=True, random_state=42, stratify=targets)

    train_subset = Subset(combined_dataset, train_idx)
    val_subset = Subset(combined_dataset, val_idx)

    train_loader = DataLoader(CDataset(train_subset, train_transforms),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=num_workers > 0)

    val_loader = DataLoader(CDataset(val_subset, val_transforms),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=num_workers > 0)

    return train_loader, val_loader


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for inputs, labels in tqdm(loader, desc='Train', colour='magenta'):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        loss.backward()
        optimizer.step()

        predicted = outputs.detach().argmax(dim=1)
        total_accuracy += (predicted == labels).sum().item()

    total_loss /= len(loader.dataset)
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', colour='magenta'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            predicted = outputs.argmax(dim=1)
            total_accuracy += (predicted == labels).sum().item()

    total_loss /= len(loader.dataset)
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy
