import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.sketchanet_classifier import SketchANet
from util.binarize import binarize
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR


def train_model(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    for inputs, labels in tqdm(loader, desc='Train', bar_format=BAR_FORMAT):
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = binarize(inputs)

        optimizer.zero_grad()
        outputs = model(inputs)

        # batch loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # calculate and update gradients
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        # batch accuracy
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        total_accuracy += accuracy

        # entire epoch loss
    total_loss /= len(loader)
    # entire epoch accuracy
    total_accuracy /= len(loader)
    return total_loss, total_accuracy


def validate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', bar_format=BAR_FORMAT):
            inputs, labels = inputs.to(device), labels.to(device)

            inputs = binarize(inputs)

            outputs = model(inputs)
            # batch loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            # batch accuracy
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            total_accuracy += accuracy

    # entire epoch loss
    total_loss /= len(loader)
    # entire epoch accuracy
    total_accuracy /= len(loader)
    return total_loss, total_accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 6

    # model initialization
    classifier = SketchANet(in_channels=3, num_classes=125)
    classifier.to(device)

    # train configuration
    batch_size = 135
    num_epochs = 250
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # data preparation
    test_data_dir = '../../datasets/Sketchy/original/test/'
    train_data_dir = '../../datasets/Sketchy/original/train/'
    val_data_dir = '../../datasets/Sketchy/original/val/'
    trained_models_dir = '../../trained_models/SketchANet'

    transform = transforms.Compose([
        transforms.RandomCrop((225, 225)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        ImageFolder(root=train_data_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        ImageFolder(root=test_data_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        ImageFolder(root=val_data_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # best model checkpoint
    best_model_checkpoint = {
        "val_loss": np.inf,
        "val_accuracy": 0.0,
        "filename": "best_model1.pth",
    }

    continue_train = {
        "from_epoch": 0,
        "from_weights": "",
    }

    early_stopping = {
        "patience": num_epochs,
        "count_epochs": 0,
    }

    # model load
    if os.path.isfile(loaded_model_path := f'{trained_models_dir}/{continue_train["from_weights"]}'):
        print(f'{FONT_COLOR}Loading model...')
        time.sleep(0.1)

        classifier.load_state_dict(torch.load(loaded_model_path))
        val_loss, val_accuracy = validate_model(classifier, val_loader, criterion, device)
        best_model_checkpoint["val_loss"] = val_loss
        best_model_checkpoint["val_accuracy"] = val_accuracy

        print(f'loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
        time.sleep(0.1)

    for epoch in range(continue_train["from_epoch"], num_epochs):

        print(f'{FONT_COLOR}\nEpoch {epoch + 1}/{num_epochs}')
        time.sleep(0.1)

        train_loss, train_accuracy = train_model(classifier, train_loader, criterion, optimizer, device)

        print(f'{FONT_COLOR}loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}')
        time.sleep(0.1)

        val_loss, val_accuracy = validate_model(classifier, val_loader, criterion, device)

        print(f'{FONT_COLOR}loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
        time.sleep(0.1)

        early_stopping["count_epochs"] += 1
        if val_loss < best_model_checkpoint["val_loss"]:
            best_model_checkpoint["val_loss"] = val_loss
            early_stopping["count_epochs"] = 0

            # save current best model
            os.makedirs(trained_models_dir, exist_ok=True)
            torch.save(classifier.state_dict(), f'{trained_models_dir}/{best_model_checkpoint["filename"]}')

        if early_stopping["count_epochs"] > early_stopping["patience"]:
            break

        time.sleep(0.1)

    print(RESET_COLOR)
