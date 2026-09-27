import json
import time
import torch
import torch.nn as nn
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from models.sketchgan_discriminator import Discriminator
from models.sketchgan_generator import Generator
from models.sketchgan_criterion import DiscriminatorLoss, GeneratorLoss
from models.resnet18_classifier import load_classifier, IMAGENET_MEAN, IMAGENET_STD
from util.dual_image_folder_dataset import DualImageFolderDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR
from util.image_functions import crop_detected_region
from util.reproducibility import set_seed


def save_checkpoint(path, generator, discriminator, gen_optim, disc_optim, epoch, gen_loss, disc_loss):
    torch.save(obj={"generator_config": generator.get_config(),
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_config": discriminator.get_config(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "gen_optim_state_dict": gen_optim.state_dict(),
                    "disc_optim_state_dict": disc_optim.state_dict(),
                    "epoch": epoch,
                    "gen_loss": gen_loss,
                    "disc_loss": disc_loss,
                    }, f=path)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        loaded_json = json.load(file)
    config = loaded_json['config']
    data = loaded_json['data']

    set_seed(config['seed'])

    # models initialization
    models_dir = Path(current_dir, config['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    generator = Generator(in_channels=1, value_range=config['value_range'])
    discriminator = Discriminator(global_shape=[1, 256, 256], local_shape=[1, 128, 128])

    generator.to(device)
    discriminator.to(device)

    # classifier model load
    classifier = load_classifier(Path(current_dir, config['classifier_to_load']), device)
    classifier.eval()
    classifier.requires_grad_(False)

    # losses initialization
    gen_criterion = GeneratorLoss(lambda1=config['lambda1'], lambda2=config['lambda2'])
    disc_criterion = DiscriminatorLoss()
    classifier_criterion = nn.CrossEntropyLoss()

    # optimizers initialisation
    gen_optim = optim.Adam(generator.parameters(), lr=config['generator_learning_rate'], betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.5, 0.999))

    # gan models load if continue train
    current_epoch = 1

    if config['continue_train']:
        checkpoint = torch.load(Path(current_dir, config['gan_to_load']), map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
        disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
        current_epoch = checkpoint['epoch'] + 1

    # data transforms initialization
    gan_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    classifier_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # data loader initialization
    loader = DataLoader(dataset=DualImageFolderDataset(first_root=Path(current_dir, data['original_dir']),
                                                       second_root=Path(current_dir, data['corrupted_dir']),
                                                       transform=gan_transform),
                        batch_size=config['batch_size'],
                        shuffle=True,
                        num_workers=config['num_workers'],
                        pin_memory=True,
                        drop_last=True)

    # tensorboard initialization
    gan_writer = SummaryWriter(f"{models_dir}/logs/train")

    # save json config
    with open(Path(models_dir, "config.json"), 'w') as file:
        json.dump(loaded_json, file, indent=4)

    # epochs initialization
    total_epochs = current_epoch - 1 + config['num_epochs']

    for _ in range(config['num_epochs']):
        print(f"{FONT_COLOR}\nEpoch {current_epoch}/{total_epochs}")
        time.sleep(0.1)

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0
        num_samples = 0

        with tqdm(loader, desc='Train', bar_format=BAR_FORMAT) as tqdm_loader:
            for original, corrupted, labels in tqdm_loader:
                original = generator.normalize(original.to(device, non_blocking=True))
                corrupted = generator.normalize(corrupted.to(device, non_blocking=True))
                labels = labels.to(device, non_blocking=True)

                generated = generator(corrupted)
                original_crop = crop_detected_region(original, corrupted, original)
                generated_crop = crop_detected_region(original, corrupted, generated)

                disc_optim.zero_grad()
                real_pred = discriminator(original, original_crop)
                fake_pred = discriminator(generated.detach(), generated_crop.detach())
                disc_loss = disc_criterion(real_pred, fake_pred)

                disc_loss.backward()
                disc_optim.step()

                gen_optim.zero_grad()
                classifier_input = classifier_transform(generator.denormalize(generated).repeat(1, 3, 1, 1))
                classifier_loss = classifier_criterion(classifier(classifier_input), labels)

                fake_pred = discriminator(generated, generated_crop)
                gen_loss = gen_criterion(original, generated, fake_pred, classifier_loss)

                gen_loss.backward()
                gen_optim.step()

                gen_epoch_loss += gen_loss.item() * labels.size(0)
                disc_epoch_loss += disc_loss.item() * labels.size(0)
                num_samples += labels.size(0)

                tqdm_loader.set_postfix({
                    f"{FONT_COLOR}Generator loss": f"{gen_loss.item():.3f}",
                    f"{FONT_COLOR}Discriminator loss": f"{disc_loss.item():.3f}"
                })

        gen_epoch_loss /= num_samples
        disc_epoch_loss /= num_samples

        print(f"{FONT_COLOR}Generator epoch loss: {gen_epoch_loss:.3f}")
        print(f"{FONT_COLOR}Discriminator epoch loss: {disc_epoch_loss:.3f}")

        gan_writer.add_scalar("Generator Loss", gen_epoch_loss, current_epoch)
        gan_writer.add_scalar("Discriminator Loss", disc_epoch_loss, current_epoch)
        gan_writer.flush()

        # saving models
        if current_epoch % config['save_every'] == 0:
            save_checkpoint(f"{models_dir}/epoch_{current_epoch}_gan.pth", generator, discriminator,
                            gen_optim, disc_optim, current_epoch, gen_epoch_loss, disc_epoch_loss)

        save_checkpoint(f"{models_dir}/last_gan.pth", generator, discriminator,
                        gen_optim, disc_optim, current_epoch, gen_epoch_loss, disc_epoch_loss)

        current_epoch += 1

    gan_writer.close()

    print(f"{RESET_COLOR}")
