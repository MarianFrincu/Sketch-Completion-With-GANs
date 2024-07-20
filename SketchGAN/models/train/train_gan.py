import os
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from models.modules.criterion import DiscriminatorLoss, GeneratorLoss
from models.sketchanet_classifier import SketchANet
from util.binarize import binarize
from util.flow_csv_dataset import CsvDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model initialization
    generator = Generator(3, 3)
    discriminator = Discriminator()
    classifier = SketchANet(3, 125)

    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    # train configuration
    batch_size = 16
    num_epochs = 20
    lr = 1e-3
    lambda1 = 100
    lambda2 = 0.5
    gen_criterion = GeneratorLoss(lambda1=lambda1, lambda2=lambda2)
    disc_criterion = DiscriminatorLoss()
    classifier_criterion = nn.CrossEntropyLoss()

    gen_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # data preparation
    csv_file_path = '../../datasets/Sketchy/train_labels.csv'
    corrupted_dir = '../../datasets/Sketchy/corrupted/train'
    original_dir = '../../datasets/Sketchy/original/train'

    transform = transforms.Compose([
        # add other augmentation techniques if necessary
        transforms.ToTensor()
    ])

    csv_dataset = CsvDataset(csv_file=csv_file_path,
                             corrupted_dir=corrupted_dir,
                             original_dir=original_dir,
                             transform=transform)

    data = DataLoader(dataset=csv_dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      # num_workers=1
                      )

    # load classifier model
    classifier_dir = "../../trained_models/SketchANet"
    classifier_weights = "best_model.pth"
    if not os.path.isfile(f'{classifier_dir}/{classifier_weights}'):
        raise FileNotFoundError(f'{classifier_dir}/{classifier_weights} is not found.')

    classifier.load_state_dict(torch.load(f'{classifier_dir}/{classifier_weights}'))

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
                classifier_transform = transforms.Compose([transforms.RandomCrop((225, 225))])
                classifier_input = classifier_transform(generated)
                classifier_input = binarize(classifier_input)

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

            gen_epoch_loss += gen_loss.item()
            disc_epoch_loss += disc_loss.item()

        print(f'{FONT_COLOR}Generator loss: {gen_epoch_loss:.3f}')
        print(f'{FONT_COLOR}Discriminator loss: {disc_epoch_loss:.3f}')

    print(f'{RESET_COLOR}')
