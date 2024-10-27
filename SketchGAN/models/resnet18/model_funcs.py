import torch
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, functional
from torch.utils.data import ConcatDataset, Subset, DataLoader

from util.custom_dataset import CustomDataset
from util.image_transforms import random_shift


def prepare_data(batch_size, num_workers, n_splits, paths):

    combined_dataset = ConcatDataset([ImageFolder(root=path) for path in paths])

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(random_shift),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_subset = Subset
    val_subset = Subset
    for fold, (train_idx, valid_idx) in enumerate(kf.split(combined_dataset)):
        train_subset = Subset(dataset=combined_dataset, indices=train_idx)
        val_subset = Subset(dataset=combined_dataset, indices=valid_idx)
        break

    train_loader = DataLoader(CustomDataset(train_subset, train_transforms),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(CustomDataset(val_subset, val_transforms),
                            batch_size=batch_size,
                            num_workers=num_workers)

    return train_loader, val_loader


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    for inputs, labels in tqdm(loader, desc='Train', colour='magenta'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # batch loss
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)

        # calculate and update gradients
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total_accuracy += (predicted == labels).sum().item()

    # entire epoch loss
    total_loss /= len(loader.dataset)
    # entire epoch accuracy
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', colour='magenta'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, dim=1)
            total_accuracy += (predicted == labels).sum().item()

    # entire epoch loss
    total_loss /= len(loader.dataset)
    # entire epoch accuracy
    total_accuracy /= len(loader.dataset)
    return total_loss, total_accuracy
