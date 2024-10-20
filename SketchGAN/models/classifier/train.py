import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import ImageFolder

from models.classifier.sketchanet import SketchANet
from util.text_format_consts import FONT_COLOR, RESET_COLOR
from util.image_transforms import normalize, binarize, random_shift
from util.custom_dataset import CustomDataset
from models.classifier.model_funcs import train_ensemble, validate_ensemble

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 8

    # ensemble of models
    ensemble = [SketchANet(in_channels=1, num_classes=250) for _ in range(4)]
    for model in ensemble:
        model.to(device)

    # train configuration
    batch_size = 135
    num_epochs = 140
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.Adam(model.parameters(), lr=lr) for model in ensemble]

    current_dir = os.path.dirname(os.path.realpath(__file__))
    trained_models_dir = os.path.abspath(os.path.join(current_dir, '../../trained_models/SketchANet'))

    # data preparation
    original_dataset_path = os.path.abspath(os.path.join(current_dir,
                                                         '../../datasets/TU_Berlin/all_images'))
    local_deformation_dataset_path = os.path.abspath(os.path.join(current_dir,
                                                                  '../../datasets/TU_Berlin/augmented/def_local'))
    global_deformation_dataset_path = os.path.abspath(os.path.join(current_dir,
                                                                   '../../datasets/TU_Berlin/augmented/def_local_global'))
    stroke_removed_dataset_path = os.path.abspath(os.path.join(current_dir,
                                                               '../../datasets/TU_Berlin/augmented/rm'))

    combined_dataset = ConcatDataset([ImageFolder(root=original_dataset_path),
                                      ImageFolder(root=local_deformation_dataset_path),
                                      ImageFolder(root=global_deformation_dataset_path),
                                      ImageFolder(root=stroke_removed_dataset_path)])

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((225, 225)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(random_shift),
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Lambda(binarize),
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((225, 225)),
        transforms.ToTensor(),
        transforms.Lambda(normalize),
        transforms.Lambda(binarize),
    ])

    # cross-validation with 3 folds
    num_folds = 3
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    streams = [torch.cuda.Stream() for _ in range(len(ensemble))]

    performance = {
        fold: {index: {'loss': [], 'accuracy': []} for index in range(len(ensemble))}
        for fold in range(num_folds)
    }

    for fold, (train_idx, valid_idx) in enumerate(kf.split(combined_dataset)):

        print(f'{FONT_COLOR}Fold {fold + 1}\n')
        train_subset = Subset(dataset=combined_dataset, indices=train_idx)
        train_loader = DataLoader(CustomDataset(train_subset, train_transforms),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

        val_subset = Subset(dataset=combined_dataset, indices=valid_idx)
        val_loader = DataLoader(CustomDataset(val_subset, val_transforms),
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True)

        for model in ensemble:
            model.reset_parameters()

            # best model checkpoint
            best_model_checkpoint = {
                "val_loss": [np.inf for _ in range(len(ensemble))],
                "folders names": ["first_network", "second_network", "third_network", "forth_network"],
            }

            for epoch in range(num_epochs):

                print(f'{FONT_COLOR}Epoch {epoch + 1}/{num_epochs}\n')

                train_losses, train_accuracies = train_ensemble(ensemble, train_loader, criterion, optimizers,
                                                                device, streams)

                for i in range(len(ensemble)):
                    print(f'{FONT_COLOR}Network {i + 1}/{len(ensemble)} '
                          f'loss: {train_losses[i]:.3f}  accuracy: {train_accuracies[i]:.3f}')

                print('\n')

                val_losses, val_accuracies = validate_ensemble(ensemble, val_loader, criterion, device, streams)

                for i in range(len(ensemble)):
                    print(f'{FONT_COLOR}Network {i + 1}/{len(ensemble)} '
                          f'loss: {val_losses[i]:.3f}  accuracy: {val_accuracies[i]:.3f}')

                    performance[fold][i]['loss'].append(val_losses[i])
                    performance[fold][i]['accuracy'].append(val_accuracies[i])

                    if val_losses[i] < best_model_checkpoint["val_loss"][i]:
                        best_model_checkpoint["val_loss"][i] = val_losses[i]

                        # save current best model
                        os.makedirs(f'{trained_models_dir}/{best_model_checkpoint["folders names"][i]}', exist_ok=True)
                        torch.save(ensemble[i].state_dict(), f'{trained_models_dir}/'
                                                             f'{best_model_checkpoint["folders names"][i]}/best_model.pth')
                        with open(
                                f'{trained_models_dir}/{best_model_checkpoint["folders names"][i]}/last_epoch.txt',
                                'w') as f:
                            f.write(f'Last epoch: {epoch}')
                print('\n\n')

        print(RESET_COLOR)

    # total evaluation for average loss and accuracy
    print(f'{FONT_COLOR}Total Performance Evaluation')

    for fold, metrics in performance.items():
        print(f'\nFold {fold}')

        for model_index, metric in metrics.items():
            avg_loss = np.mean(metric['loss'])
            avg_accuracy = np.mean(metric['accuracy'])

            print(f'Model {model_index} - Avg Loss: {avg_loss:.3f}, Avg Accuracy: {avg_accuracy:.3f}')
