import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from models.modules.criterion import DiscriminatorLoss, GeneratorLoss
from models.sketchanet_classifier import SketchANet
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
    # to do
    data = DataLoader

    # load model
    # to do

    generator.train()
    discriminator.train()
    classifier.eval()

    for epoch in range(num_epochs):
        print(f'{FONT_COLOR}\nEpoch {epoch + 1}/{num_epochs}')
        time.sleep(0.1)

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for (original, corrupted, labels) in tqdm(data, desc='Train', bar_format=BAR_FORMAT):
            original, corrupted, labels = original.to(device), corrupted.to(device), labels.to(device)

            generated = generator(corrupted)
            fake_pred = discriminator(corrupted, nn.functional.interpolate(generated, size=128))

            with torch.no_grad():
                predicted_labels = classifier(generated)
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
