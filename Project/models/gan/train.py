import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

import torchvision
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from models.gan.discriminator import Discriminator
from models.gan.generator import Generator
from models.gan.modules.criterion import DiscriminatorLoss, GeneratorLoss
from util.dual_image_folder_dataset import DualImageFolderDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR
from util.image_functions import crop_detected_region

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        loaded_json = json.load(file)
    config = loaded_json['config']
    data = loaded_json['data']

    # models initialization
    models_dir = Path(current_dir, config['models_dir'])

    generator = Generator(in_channels=1, out_channels=1)
    discriminator = Discriminator(global_shape=[1, 256, 256], local_shape=[1, 128, 128])
    classifier = resnet18()
    classifier.fc = nn.Linear(classifier.fc.in_features, out_features=config['classifier_classes'])

    generator.to(device)
    discriminator.to(device)
    classifier.to(device)

    # classifier model load
    classifier.load_state_dict(torch.load(Path(current_dir, config['classifier_to_load']),
                                          weights_only=True,
                                          map_location=device
                                          ))

    # gan models load if continue train
    current_epoch = 1
    gen_best_loss = np.inf

    if Path(current_dir, config['gan_to_load']).is_file():
        print(True)

    if config['continue_train']:
        checkpoint = torch.load(Path(current_dir, config['gan_to_load']), map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        current_epoch = checkpoint['epoch'] + 1
        gen_best_loss = checkpoint['gen_loss']

    # losses initialization
    gen_criterion = GeneratorLoss(lambda1=config['lambda1'], lambda2=config['lambda2'])
    disc_criterion = DiscriminatorLoss()
    classifier_criterion = nn.CrossEntropyLoss()

    # optimizers initialisation
    gen_optim = optim.Adam(generator.parameters(), lr=config['generator_learning_rate'], betas=(0.5, 0.999))
    disc_optim = optim.Adam(discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.5, 0.999))

    # data transforms initialization
    gan_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    classifier_transform = transforms.Compose([
        transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1)),
        transforms.Resize((224, 224))
    ])

    # data loader initialization
    loader = DataLoader(dataset=DualImageFolderDataset(first_root=Path(current_dir, data['original_dir']),
                                                       second_root=Path(current_dir, data['corrupted_dir']),
                                                       transform=gan_transform),
                        batch_size=config['batch_size'],
                        shuffle=True,
                        num_workers=10)

    # tensorboard initialization
    gan_writer = SummaryWriter(f"{models_dir}/logs/train")

    # save json config
    with open(Path(models_dir, "config.json"), 'w') as file:
        json.dump(loaded_json, file)

    # epochs initialization
    total_epochs = current_epoch - 1 + config['num_epochs']

    counter = 10  # used for saving the model each n epochs

    for _ in range(config['num_epochs']):
        print(f"{FONT_COLOR}\nEpoch {current_epoch}/{total_epochs}")
        time.sleep(0.1)

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        with tqdm(loader, desc='Train', bar_format=BAR_FORMAT) as tqdm_loader:
            for i, (original, corrupted, labels) in enumerate(tqdm_loader):
                original, corrupted, labels = original.to(device), corrupted.to(device), labels.to(device)

                generated = generator(corrupted)
                original_crop = crop_detected_region(original, corrupted, original)
                generated_crop = crop_detected_region(original, corrupted, generated)

                disc_optim.zero_grad()
                fake_pred = discriminator(corrupted, generated_crop.detach())
                real_pred = discriminator(corrupted, original_crop)
                disc_loss = disc_criterion(real_pred, fake_pred)

                disc_loss.backward()
                disc_optim.step()

                gen_optim.zero_grad()
                predicted_labels = classifier(torch.stack([classifier_transform(img) for img in generated.detach()]))
                classifier_loss = classifier_criterion(predicted_labels, labels)

                fake_pred = discriminator(corrupted, generated_crop)
                gen_loss = gen_criterion(original, generated, fake_pred, classifier_loss)

                gen_loss.backward()
                gen_optim.step()

                gen_epoch_loss += gen_loss.item() * labels.size(0)
                disc_epoch_loss += disc_loss.item() * labels.size(0)

                tqdm_loader.set_postfix({
                    f"{FONT_COLOR}Generator loss": f"{gen_loss.item():.3f}",
                    f"{FONT_COLOR}Discriminator loss": f"{disc_loss.item():.3f}"
                })

        gen_epoch_loss /= len(loader.dataset)
        disc_epoch_loss /= len(loader.dataset)

        print(f"{FONT_COLOR}Generator epoch loss: {gen_epoch_loss:.3f}")
        print(f"{FONT_COLOR}Discriminator epoch loss: {disc_epoch_loss:.3f}")

        gan_writer.add_scalar("Generator Loss", gen_epoch_loss, current_epoch)
        gan_writer.add_scalar("Discriminator Loss", disc_epoch_loss, current_epoch)

        generated = generator(corrupted)
        gan_writer.add_image(tag="{=} Generated <==> Corrupted <==> Original {=}",
                             img_tensor=torchvision.utils.make_grid(
                                 torch.stack([generated[0].detach(), corrupted[0], original[0]]).cpu(),
                                 nrow=3,
                                 normalize=True),
                             global_step=current_epoch
                             )
        gan_writer.flush()

        # saving models
        if current_epoch % counter == 0:
            torch.save(obj={"generator_state_dict": generator.state_dict(),
                            "discriminator_state_dict": discriminator.state_dict(),
                            "epoch": current_epoch,
                            "gen_loss": gen_epoch_loss,
                            "disc_loss": disc_epoch_loss,
                            }, f=f"{models_dir}/epoch_{current_epoch}_gan.pth")

        torch.save(obj={"generator_state_dict": generator.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "epoch": current_epoch,
                        "gen_loss": gen_epoch_loss,
                        "disc_loss": disc_epoch_loss,
                        }, f=f"{models_dir}/last_gan.pth")

        if gen_epoch_loss <= gen_best_loss:
            gen_best_loss = gen_epoch_loss
            torch.save(obj={"generator_state_dict": generator.state_dict(),
                            "discriminator_state_dict": discriminator.state_dict(),
                            "epoch": current_epoch,
                            "gen_loss": gen_epoch_loss,
                            "disc_loss": disc_epoch_loss,
                            }, f=f"{models_dir}/best_gan.pth")

        current_epoch += 1

    gan_writer.close()

    print(f"{RESET_COLOR}")
