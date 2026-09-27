import json
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from resnet18_classifier_train.model_funcs import train_model, validate_model, prepare_data
from models.resnet18_classifier import Resnet18Classifier
from util.reproducibility import set_seed


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    torch.save({"state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": model.num_classes,
                "epoch": epoch,
                "best_loss": best_loss},
               path)


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        loaded_json = json.load(file)
    config = loaded_json['config']
    data = loaded_json['data']

    set_seed(config['seed'])

    model_dir = Path(current_dir, config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    resnet = Resnet18Classifier(config['num_classes'])
    resnet.freeze_backbone(config['freeze'])
    resnet.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()))

    current_epoch = 1
    best_loss = np.inf

    if config['continue_train']:
        checkpoint = torch.load(Path(current_dir, config['model_to_load']), map_location=device, weights_only=True)
        resnet.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    train_loader, val_loader = prepare_data(batch_size=config['batch_size'],
                                            num_workers=config['num_workers'],
                                            train_size=config['train_size'],
                                            paths=[Path(current_dir, path) for path in data])

    train_writer = SummaryWriter(f"{model_dir}/logs/train")
    val_writer = SummaryWriter(f"{model_dir}/logs/val")

    with open(Path(model_dir, "config.json"), 'w') as file:
        json.dump(loaded_json, file, indent=4)

    total_epochs = current_epoch - 1 + sum(config['epochs'])

    for num_epochs, learning_rate, weight_decay in zip(config['epochs'], config['learning_rates'], config['weight_decays']):

        optimizer.param_groups[0]['lr'] = learning_rate
        optimizer.param_groups[0]['weight_decay'] = weight_decay

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

            if val_loss < best_loss:
                best_loss = val_loss

                save_checkpoint(f"{model_dir}/best_model.pth", resnet, optimizer, current_epoch, best_loss)
                with open(f"{model_dir}/best_epoch.txt", 'w') as f:
                    f.write(f"Best model epoch: {current_epoch}")

            save_checkpoint(f"{model_dir}/last_model.pth", resnet, optimizer, current_epoch, best_loss)
            with open(f"{model_dir}/last_epoch.txt", 'w') as f:
                f.write(f"Last epoch: {current_epoch}")

            current_epoch += 1

            time.sleep(0.1)

    train_writer.close()
    val_writer.close()
