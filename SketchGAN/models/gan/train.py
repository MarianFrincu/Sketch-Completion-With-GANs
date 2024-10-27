import os
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from models.gan.discriminator import Discriminator
from models.gan.generator import Generator
from models.gan.modules.criterion import DiscriminatorLoss, GeneratorLoss
from util.flow_csv_dataset import CsvDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 10

    # model initialization
    generator = Generator(3, 3)
    discriminator = Discriminator()
    classifier = resnet18()
    classifier.fc = nn.Linear(classifier.fc.in_features, 125)

    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    # train configuration
    batch_size = 16
    num_epochs = 100
    lr = 1e-3
    lambda1 = 100
    lambda2 = 0.5
    gen_criterion = GeneratorLoss(lambda1=lambda1, lambda2=lambda2)
    disc_criterion = DiscriminatorLoss()
    classifier_criterion = nn.CrossEntropyLoss()

    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # data preparation
    csv_file_path = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/sketchy_labels.csv'))
    corrupted_dir = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/corrupted/all_images'))
    original_dir = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/original/all_images'))

    transform = transforms.Compose([
        # add other augmentation techniques if necessary
        transforms.ToTensor()
    ])

    data = DataLoader(dataset=CsvDataset(csv_file_path, corrupted_dir, original_dir, transform),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)

    # load classifier model
    classifier_dir = os.path.abspath(os.path.join(current_dir, "../../models/classifier/resnet18/trained_models/resnet18_1e-4_finetune_sketchy/best_model.pth"))
    if not os.path.isfile(f'{classifier_dir}'):
        raise FileNotFoundError(f'{classifier_dir} is not found.')

    classifier.load_state_dict(torch.load(f'{classifier_dir}'))

    generator.train()
    discriminator.train()
    classifier.eval()
    for epoch in range(num_epochs):
        print(f'{FONT_COLOR}\nEpoch {epoch + 1}/{num_epochs}')
        time.sleep(0.1)

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for (corrupted, original, labels) in tqdm(data, desc='Train', bar_format=BAR_FORMAT):
            corrupted, original, labels = corrupted.to(device), original.to(device), labels.to(device)

            generated = generator(corrupted)
            fake_pred = discriminator(corrupted, nn.functional.interpolate(generated, size=128))

            with torch.no_grad():
                # preprocess generator output data and do inference on classifier model
                classifier_input = nn.functional.interpolate(generated.detach(), size=128)

                predicted_labels = classifier(classifier_input)
                classifier_loss = classifier_criterion(predicted_labels, labels)

            gen_loss = gen_criterion(original, generated, fake_pred, classifier_loss)

            fake_pred = discriminator(corrupted, nn.functional.interpolate(generated.detach(), size=128))
            real_pred = discriminator(corrupted, nn.functional.interpolate(original, size=128))
            disc_loss = disc_criterion(fake_pred, real_pred)

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            gen_epoch_loss += gen_loss.item() * labels.size(0)
            disc_epoch_loss += disc_loss.item() * labels.size(0)

        print(f'{FONT_COLOR}Generator loss: {gen_epoch_loss / len(data):.3f}')
        print(f'{FONT_COLOR}Discriminator loss: {disc_epoch_loss / len(data):.3f}')

    print(f'{RESET_COLOR}')
