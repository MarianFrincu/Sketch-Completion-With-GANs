import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import numpy as np
import time

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model initialization
model = SketchANet(in_channels=3, num_classes=125)
model.to(device)

# train configuration
batch_size = 512
num_epochs = 250
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# data preparation
test_data_dir = '../../datasets/Sketchy/original/test/'
train_data_dir = '../../datasets/Sketchy/original/train/'
val_data_dir = '../../datasets/Sketchy/original/val/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop((225, 225)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
])

train_dataset = ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

val_dataset = ImageFolder(root=val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

# best model checkpoint
best_val_loss = np.inf
best_val_accuracy = 0.0
patience = 10
count_epochs = 0
trained_models_dir = "../../trained_models/SketchANet"
best_model_filename = "best_model.pth"
loaded_model_filename = ""

print(FONT_COLOR)

# model load
if os.path.isfile(f'{trained_models_dir}/{loaded_model_filename}'):
    print('Loading model...')
    time.sleep(0.1)

    model.load_state_dict(torch.load(f'{trained_models_dir}/{loaded_model_filename}'))
    best_val_loss, best_val_accuracy = validate_model(model, val_loader, criterion, device)

for epoch in range(num_epochs):

    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    time.sleep(0.1)

    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)

    print(f'loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}')
    time.sleep(0.1)

    val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

    print(f'loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
    time.sleep(0.1)

    count_epochs += 1
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        count_epochs = 0
        os.makedirs(trained_models_dir, exist_ok=True)
        torch.save(model.state_dict(), f'{trained_models_dir}/{best_model_filename}')

    if count_epochs > patience:
        break

    time.sleep(0.1)

print(RESET_COLOR)
