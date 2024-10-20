import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from util.text_format_consts import BAR_FORMAT


def train_model(model, criterion, optimizer, inputs, labels, stream):
    with torch.cuda.stream(stream):
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        batch_loss = loss.item() * labels.size(0)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        batch_accuracy = (predicted == labels).sum().item()
        return batch_loss, batch_accuracy


def train_ensemble(ensemble, loader, criterion, optimizers, device, streams):
    for model in ensemble:
        model.train()

    total_losses = [0.0 for _ in range(len(ensemble))]
    total_accuracies = [0.0 for _ in range(len(ensemble))]

    for inputs, labels in tqdm(loader, desc='Train', bar_format=BAR_FORMAT):
        inputs, labels = inputs.to(device), labels.to(device)

        futures = []

        with ThreadPoolExecutor() as executor:
            for i in range(len(ensemble)):
                futures.append(executor.submit(train_model, ensemble[i], criterion, optimizers[i], inputs, labels,
                                               streams[i]))

            for i in range(len(futures)):
                batch_loss, batch_accuracy = futures[i].result()
                total_losses[i] += batch_loss
                total_accuracies[i] += batch_accuracy

    # entire epoch loss
    total_losses = [loss / len(loader.dataset) for loss in total_losses]
    # entire epoch accuracy
    total_accuracies = [accuracy / len(loader.dataset) for accuracy in total_accuracies]
    return total_losses, total_accuracies


def validate_model(model, criterion, inputs, labels, stream):
    with torch.cuda.stream(stream):
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        batch_loss = loss.item() * labels.size(0)

        _, predicted = torch.max(outputs.data, dim=1)
        batch_accuracy = (predicted == labels).sum().item()
        return batch_loss, batch_accuracy


def validate_ensemble(ensemble, loader, criterion, device, streams):
    for model in ensemble:
        model.eval()

    total_losses = [0.0 for _ in range(len(ensemble))]
    total_accuracies = [0.0 for _ in range(len(ensemble))]

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', bar_format=BAR_FORMAT):
            inputs, labels = inputs.to(device), labels.to(device)

            futures = []

            with ThreadPoolExecutor() as executor:
                for i in range(len(ensemble)):
                    futures.append(
                        executor.submit(validate_model, ensemble[i], criterion, inputs, labels, streams[i]))

                for i in range(len(futures)):
                    batch_loss, batch_accuracy = futures[i].result()
                    total_losses[i] += batch_loss
                    total_accuracies[i] += batch_accuracy

    # entire epoch loss
    total_losses = [loss / len(loader.dataset) for loss in total_losses]
    # entire epoch accuracy
    total_accuracies = [accuracy / len(loader.dataset) for accuracy in total_accuracies]
    return total_losses, total_accuracies
