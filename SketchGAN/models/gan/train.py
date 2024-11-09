import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from models.gan.discriminator import Discriminator
from models.gan.generator import Generator
from models.gan.modules.criterion import DiscriminatorLoss, GeneratorLoss
from util.flow_csv_dataset import CsvDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR
from util.image_transforms import crop_detected_region

from torchvision.transforms import ToPILImage


def save_img(tensor, path):
    to_pil = ToPILImage()
    image = to_pil(tensor)

    image.save(path)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 10

    # model initialization
    generator = Generator(in_channels=3, out_channels=3)
    discriminator = Discriminator()
    classifier = resnet18()
    classifier.fc = nn.Linear(classifier.fc.in_features, 125)

    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    # train configuration
    batch_size = 16
    num_epochs = 20
    lr_gen = 2e-4
    lr_disc = 1e-4
    lambda1 = 100
    lambda2 = 0.5
    gen_criterion = GeneratorLoss(lambda1=lambda1, lambda2=lambda2)
    disc_criterion = DiscriminatorLoss()
    classifier_criterion = nn.CrossEntropyLoss()

    gen_optim = optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

    current_dir = os.path.dirname(os.path.realpath(__file__))

    # data preparation
    csv_file_path = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/sketchy_labels.csv'))
    corrupted_dir = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/corrupted/all_images'))
    original_dir = os.path.abspath(os.path.join(current_dir, '../../datasets/Sketchy/original/all_images'))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data = DataLoader(dataset=CsvDataset(csv_file_path, corrupted_dir, original_dir, transform),
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)

    # load classifier model
    classifier_dir = os.path.abspath(
        os.path.join(current_dir, "../../trained_models/resnet18_finetune_Sketchy3/best_model.pth"))
    if not os.path.isfile(f'{classifier_dir}'):
        raise FileNotFoundError(f'{classifier_dir} is not found.')

    classifier.load_state_dict(torch.load(f'{classifier_dir}', weights_only=True))

    generator.train()
    discriminator.train()
    classifier.eval()

    gen_best_loss = np.inf

    model_dir = os.path.abspath(os.path.join(current_dir, '../../trained_models/GAN'))

    gan_writer = SummaryWriter(f'{model_dir}/logs/train')

    for epoch in range(num_epochs):
        print(f'{FONT_COLOR}\nEpoch {epoch + 1}/{num_epochs}')
        time.sleep(0.1)

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        with tqdm(data, desc='Train', bar_format=BAR_FORMAT) as tqdm_bar:
            for i, batch in enumerate(tqdm_bar):
                corrupted, original, labels = batch
                corrupted, original, labels = corrupted.to(device), original.to(device), labels.to(device)

                generated = generator(corrupted)
                original_crop = crop_detected_region(original, corrupted, original)
                generated_crop = crop_detected_region(original, corrupted, generated)

                save_img(generated[0], f'{model_dir}/epoch_{epoch + 1}/generated.png')
                save_img(original_crop[0], f'{model_dir}/epoch_{epoch + 1}/original_crop.png')
                save_img(generated_crop[0], f'{model_dir}/epoch_{epoch + 1}/generated_crop.png')
                save_img(original[0], f'{model_dir}/epoch_{epoch + 1}/original.png')
                save_img(corrupted[0], f'{model_dir}/epoch_{epoch + 1}/corrupted.png')

                disc_optim.zero_grad()
                fake_pred = discriminator(corrupted, generated_crop.detach())
                real_pred = discriminator(corrupted, original_crop)
                disc_loss = disc_criterion(real_pred, fake_pred)

                disc_loss.backward()
                disc_optim.step()

                gen_optim.zero_grad()
                predicted_labels = classifier(nn.functional.interpolate(generated, size=224))
                classifier_loss = classifier_criterion(predicted_labels, labels)

                fake_pred = discriminator(corrupted, generated_crop)
                gen_loss = gen_criterion(original, generated, fake_pred, classifier_loss)

                gen_loss.backward()
                gen_optim.step()

                gen_epoch_loss += gen_loss.item() * labels.size(0)
                disc_epoch_loss += disc_loss.item() * labels.size(0)

                tqdm_bar.set_postfix({
                    f'{FONT_COLOR}Generator loss': f'{gen_loss.item():.3f}',
                    f'{FONT_COLOR}Discriminator loss': f'{disc_loss.item():.3f}'
                })

        gen_epoch_loss /= len(data.dataset)
        disc_epoch_loss /= len(data.dataset)

        print(f'{FONT_COLOR}Generator epoch loss: {gen_epoch_loss:.3f}')
        print(f'{FONT_COLOR}Discriminator epoch loss: {disc_epoch_loss:.3f}')

        gan_writer.add_scalar('Generator Loss', gen_epoch_loss, epoch)
        gan_writer.add_scalar('Discriminator Loss', disc_epoch_loss, epoch)
        gan_writer.flush()

        torch.save(generator.state_dict(), f'{model_dir}/last_generator.pth')
        torch.save(discriminator.state_dict(), f'{model_dir}/last_discriminator.pth')
        with open(f'{model_dir}/last_epoch.txt', 'w') as f:
            f.write(f'Last epoch: {epoch + 1}')

        if gen_epoch_loss <= gen_best_loss:
            gen_best_loss = gen_epoch_loss
            torch.save(generator.state_dict(), f'{model_dir}/best_generator.pth')
            torch.save(discriminator.state_dict(), f'{model_dir}/best_discriminator.pth')
            with open(f'{model_dir}/best_epoch.txt', 'w') as f:
                f.write(f'Best epoch: {epoch + 1}')

    print(f'{RESET_COLOR}')
