import json
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

from models.resnet18_classifier import load_classifier, IMAGENET_MEAN, IMAGENET_STD
from resnet18_classifier_train.model_funcs import validate_model

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['evaluate']

    resnet = load_classifier(Path(current_dir, config['model_to_evaluate']), device)

    criterion = nn.CrossEntropyLoss()

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    datasets = [ImageFolder(root=Path(current_dir, path), transform=test_transforms) for path in config['data']]

    if any(dataset.classes != datasets[0].classes for dataset in datasets):
        raise ValueError("All data folders must contain the same classes")

    combined_dataset = ConcatDataset(datasets)

    test_loader = DataLoader(combined_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True)

    test_loss, test_accuracy = validate_model(resnet, test_loader, criterion, device)

    print(f'loss: {test_loss:.3f}  accuracy: {test_accuracy:.3f}')
