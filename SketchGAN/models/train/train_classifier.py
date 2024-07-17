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

train_data_dir = '../../datasets/Sketchy/images/original_split/train/'
val_data_dir = '../../datasets/Sketchy/images/original_split/val/'

transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root=train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = ImageFolder(root=val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = SketchANet(in_channels=3, num_classes=125)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
best_val_loss = np.Inf

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc='Epoch {}/{}'.format(epoch + 1, num_epochs), colour="magenta"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f'Train Loss: {train_loss}')

    time.sleep(0.1)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Epoch {}/{}'.format(epoch + 1, num_epochs), colour="magenta"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '../../trained_models/classifier/best_model.pth')
        print(f'New best model saved with validation loss: {best_val_loss}')

    time.sleep(0.1)
