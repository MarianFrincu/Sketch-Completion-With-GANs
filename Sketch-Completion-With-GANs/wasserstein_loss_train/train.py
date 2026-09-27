import json
import time
import torch

from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from models.critic import Critic
from models.sketchgan_generator import Generator
from models.wasserstein_criterion import WCriticLoss, WGeneratorLoss
from models.gradient_penalty import gradient_penalty
from util.dual_image_folder_dataset import DualImageFolderDataset
from util.text_format_consts import FONT_COLOR, BAR_FORMAT, RESET_COLOR
from util.reproducibility import set_seed


def save_checkpoint(path, generator, critic, gen_optim, critic_optim, epoch, gen_loss, critic_loss):
    torch.save(obj={"generator_config": generator.get_config(),
                    "generator_state_dict": generator.state_dict(),
                    "critic_config": critic.get_config(),
                    "critic_state_dict": critic.state_dict(),
                    "gen_optim_state_dict": gen_optim.state_dict(),
                    "critic_optim_state_dict": critic_optim.state_dict(),
                    "epoch": epoch,
                    "gen_loss": gen_loss,
                    "critic_loss": critic_loss,
                    }, f=path)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    critic = Critic()

    generator.to(device)
    critic.to(device)

    # losses initialization
    gen_criterion = WGeneratorLoss()
    critic_criterion = WCriticLoss(lambda_gp=config['lambda_gp'])

    # optimizers initialisation
    gen_optim = optim.Adam(generator.parameters(), lr=config['generator_learning_rate'], betas=(0.0, 0.9))
    critic_optim = optim.Adam(critic.parameters(), lr=config['critic_learning_rate'], betas=(0.0, 0.9))

    # gan models load if continue train
    current_epoch = 1

    if config['continue_train']:
        checkpoint = torch.load(Path(current_dir, config['gan_to_load']), map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state_dict'])
        critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        current_epoch = checkpoint['epoch'] + 1

    # data transforms initialization
    gan_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
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
        gen_steps = 0
        critic_epoch_loss = 0.0
        num_samples = 0

        with tqdm(loader, desc='Train', bar_format=BAR_FORMAT) as tqdm_loader:
            for i, (original, corrupted, _) in enumerate(tqdm_loader):
                original = generator.normalize(original.to(device, non_blocking=True))
                corrupted = generator.normalize(corrupted.to(device, non_blocking=True))

                with torch.no_grad():
                    generated = generator(corrupted)

                critic_optim.zero_grad()
                real_scores = critic(original)
                fake_scores = critic(generated)
                critic_loss = critic_criterion(real_scores, fake_scores,
                                               gradient_penalty(critic, original, generated))

                critic_loss.backward()
                critic_optim.step()

                if i % config['n_critic'] == 0:
                    gen_optim.zero_grad()
                    generated = generator(corrupted)

                    fake_scores = critic(generated)
                    gen_loss = gen_criterion(fake_scores)

                    gen_loss.backward()
                    gen_optim.step()

                    gen_epoch_loss += gen_loss.item() * original.size(0)
                    gen_steps += original.size(0)

                critic_epoch_loss += critic_loss.item() * original.size(0)
                num_samples += original.size(0)

                tqdm_loader.set_postfix({
                    f"{FONT_COLOR}Generator loss": f"{gen_loss.item():.3f}",
                    f"{FONT_COLOR}Critic loss": f"{critic_loss.item():.3f}"
                })

        critic_epoch_loss /= num_samples
        gen_epoch_loss /= gen_steps

        print(f"{FONT_COLOR}Generator epoch loss: {gen_epoch_loss:.3f}")
        print(f"{FONT_COLOR}Critic epoch loss: {critic_epoch_loss:.3f}")

        gan_writer.add_scalar("Generator Loss", gen_epoch_loss, current_epoch)
        gan_writer.add_scalar("Critic Loss", critic_epoch_loss, current_epoch)
        gan_writer.flush()

        # saving models
        if current_epoch % config['save_every'] == 0:
            save_checkpoint(f"{models_dir}/epoch_{current_epoch}_gan.pth", generator, critic,
                            gen_optim, critic_optim, current_epoch, gen_epoch_loss, critic_epoch_loss)

        save_checkpoint(f"{models_dir}/last_gan.pth", generator, critic,
                        gen_optim, critic_optim, current_epoch, gen_epoch_loss, critic_epoch_loss)

        current_epoch += 1

    gan_writer.close()

    print(f"{RESET_COLOR}")
