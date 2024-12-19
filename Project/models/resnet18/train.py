import json
import time
import numpy as np
import torch
import torchvision
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from models.resnet18.model_funcs import train_model, validate_model, prepare_data

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        loaded_json = json.load(file)
    config = loaded_json['config']
    data = loaded_json['data']

    # model initialization
    model_dir = Path(current_dir, config['model_dir'])

    resnet = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, config['num_classes'])

    for param in resnet.parameters():
        param.requires_grad = not config['freeze']

    for param in resnet.fc.parameters():
        param.requires_grad = True

    resnet.to(device)

    current_epoch = 1
    best_loss = np.inf

    if config['continue_train']:
        checkpoint = torch.load(Path(current_dir, config['model_to_load']), map_location=device, weights_only=True)
        resnet.load_state_dict(checkpoint['state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    # loss initialization
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer initialization
    optimizer = torch.optim.Adam(resnet.parameters())

    # data loader initialization
    data_paths = [Path(current_dir, path) for path in data]

    train_loader, val_loader = prepare_data(batch_size=config['batch_size'],
                                            num_workers=10,
                                            n_splits=config['num_folds'],
                                            paths=data_paths)

    # tensorboard initialization
    train_writer = SummaryWriter(f"{model_dir}/logs/train")
    val_writer = SummaryWriter(f"{model_dir}/logs/val")

    # save json config
    with open(Path(model_dir, "config.json"), 'w') as file:
        json.dump(loaded_json, file)

    # epochs initialization
    total_epochs = current_epoch - 1 + sum(config['epochs'])

    for num_epochs, learning_rate, weight_decay in zip(config['epochs'], config['learning_rates'], config['weight_decays']):

        optimizer.lr = learning_rate
        optimizer.weight_decay = weight_decay

        for _ in range(num_epochs):
            print(f"\nEpoch {current_epoch}/{total_epochs}")
            time.sleep(0.1)

            train_loss, train_accuracy = train_model(resnet, train_loader, criterion, optimizer, device)

            train_writer.add_scalar("loss", train_loss, current_epoch)
            train_writer.add_scalar("accuracy", train_accuracy, current_epoch)

            print(f"loss: {train_loss:.3f}  accuracy: {train_accuracy:.3f}")
            time.sleep(0.1)

            val_loss, val_accuracy = validate_model(resnet, val_loader, criterion, device)

            val_writer.add_scalar("loss", val_loss, current_epoch)
            val_writer.add_scalar("accuracy", val_accuracy, current_epoch)

            print(f"loss: {val_loss:.3f}  accuracy: {val_accuracy:.3f}")
            time.sleep(0.1)

            # save last trained model
            torch.save(resnet.state_dict(), f"{model_dir}/last_model.pth")
            with open(f"{model_dir}/last_epoch.txt", 'w') as f:
                f.write(f"Last epoch: {current_epoch}")

            # save current best model
            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(resnet.state_dict(), f"{model_dir}/best_model.pth")
                with open(f"{model_dir}/best_epoch.txt", 'w') as f:
                    f.write(f"Best model epoch: {current_epoch}")

            current_epoch += 1

            time.sleep(0.1)

    train_writer.close()
    val_writer.close()
