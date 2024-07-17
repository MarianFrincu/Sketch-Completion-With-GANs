import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import time

from models.sketchanet_classifier import SketchANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SketchANet(in_channels=3, num_classes=125)
model.to(device)

# train configuration
train_data_dir = '../../datasets/Sketchy/images/original_split/train/'
val_data_dir = '../../datasets/Sketchy/images/original_split/val/'
batch_size = 32
num_epochs = 100
lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# data preparation
transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageFolder(root=val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# best model checkpoint
best_val_loss = np.Inf
patience = 10
count_epochs = 0
trained_models_dir = "../../trained_models/classifier"
best_model_filename = "best_model2.pth"

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    time.sleep(0.1)

    model.train()
    train_loss = 0.0
    correct_train = 0
    for inputs, labels in tqdm(train_loader, desc='Train', colour="magenta"):
        inputs, labels = inputs.to(device), labels.to(device)

        # binarize inputs
        inputs[inputs < 1.] = 0.
        inputs = 1. - inputs

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item() / labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = correct_train / len(train_loader)

    print(f'loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}')
    time.sleep(0.1)

    model.eval()
    val_loss = 0.0
    correct_val = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', colour="magenta"):
            inputs, labels = inputs.to(device), labels.to(device)

            # binarize inputs
            inputs[inputs < 1.] = 0.
            inputs = 1. - inputs

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item() / labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_val / len(val_loader)

    print(f'loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
    time.sleep(0.1)

    count_epochs += 1
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        count_epochs = 0
        torch.save(model.state_dict(), f'{trained_models_dir}/{best_model_filename}')

    if count_epochs > patience:
        break

    time.sleep(0.1)
