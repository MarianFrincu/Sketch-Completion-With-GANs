import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torch.utils.data import DataLoader, ConcatDataset

from util.c_dataset import CDataset
from resnet18_classifier_train.model_funcs import validate_model

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_workers = 10
    batch_size = 135

    resnet = resnet18()
    resnet.fc = nn.Linear(resnet.fc.in_features, 125)

    criterion = nn.CrossEntropyLoss()

    model_to_load = "../trained_models/resnet18_finetune_Sketchy/best_model.pth"

    paths = ["../datasets/Sketchy_split/test"]

    combined_dataset = ConcatDataset([ImageFolder(root=path) for path in paths])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_loader = DataLoader(CDataset(combined_dataset, test_transforms),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    if os.path.isfile(f'{model_to_load}'):
        resnet.load_state_dict(torch.load(f'{model_to_load}', weights_only=True))
        resnet.to(device)

        test_loss, test_accuracy = validate_model(resnet, test_loader, criterion, device)

        print(f'loss: {test_loss:.3f}  accuracy: {test_accuracy:.3f}')

    else:
        print('No model found')
