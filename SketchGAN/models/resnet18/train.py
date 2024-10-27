import os
import time
import json
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from models.resnet18.model_funcs import train_model, validate_model, prepare_data

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.abspath('config.json'), 'r') as file:
        loaded_json = json.load(file)

    config = loaded_json['config']
    data = loaded_json['data']

    num_workers = config['num_workers']
    current_epoch = config['initial_epoch']
    epochs = config['epochs']
    learning_rates = config['learning_rates']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    num_folds = config['num_folds']
    num_classes = config['num_classes']
    freeze = config['freeze']

    data_paths = [os.path.abspath(f'../../{path}') for path in data]

    model_dir = config['model_dir']
    model_dir = os.path.abspath(f'../../{model_dir}')

    # model declaration
    resnet = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

    for param in resnet.parameters():
        param.requires_grad = not freeze

    for param in resnet.fc.parameters():
        param.requires_grad = True

    if config['continue_train']:
        model_to_load = os.path.join(model_dir, config['model_to_load'])
        if os.path.isfile(model_to_load):
            resnet.load_state_dict(model_to_load)

    resnet.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), weight_decay=weight_decay)

    train_loader, val_loader = prepare_data(batch_size=batch_size,
                                            num_workers=num_workers,
                                            n_splits=num_folds,
                                            paths=data_paths)

    # tensorboard initialization
    train_writer = SummaryWriter(f'{model_dir}/logs/train')
    val_writer = SummaryWriter(f'{model_dir}/logs/val')

    best_loss = np.inf

    with open(f'{model_dir}/config.json', 'w') as file:
        json.dump(loaded_json, file, indent=4)

    total_epochs = sum(epochs)

    for num_epochs, learning_rate in zip(epochs, learning_rates):

        optimizer.lr = learning_rate

        for _ in range(num_epochs):
            print(f'\nEpoch {current_epoch}/{total_epochs}')
            time.sleep(0.1)

            train_loss, train_accuracy = train_model(resnet, train_loader, criterion, optimizer, device)

            train_writer.add_scalar('loss', train_loss, current_epoch)
            train_writer.add_scalar('accuracy', train_accuracy, current_epoch)

            print(f'loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}')
            time.sleep(0.1)

            val_loss, val_accuracy = validate_model(resnet, val_loader, criterion, device)

            val_writer.add_scalar('loss', val_loss, current_epoch)
            val_writer.add_scalar('accuracy', val_accuracy, current_epoch)

            print(f'loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}')
            time.sleep(0.1)

            # save last trained model
            torch.save(resnet.state_dict(), f'{model_dir}/last_model.pth')
            with open(f'{model_dir}/last_epoch.txt', 'w') as f:
                f.write(f'Last epoch: {current_epoch}')

            # save current best model
            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(resnet.state_dict(), f'{model_dir}/best_model.pth')
                with open(f'{model_dir}/best_epoch.txt', 'w') as f:
                    f.write(f'Best model epoch: {current_epoch}')

            current_epoch += 1

            time.sleep(0.1)

        train_writer.close()
        val_writer.close()
