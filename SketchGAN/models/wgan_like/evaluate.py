import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.wgan_like.generator import Generator
from util.dual_image_folder_dataset import DualImageFolderDataset
from util.text_format_consts import BAR_FORMAT

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gen = Generator(in_channels=1, out_channels=1)

    gen.to(device)

    gan_save = torch.load('../../trained_models/GrayscaleWGANTEST/last_gan.pth', weights_only=True, map_location=device)

    gen.load_state_dict(gan_save['generator_state_dict'])
    gen.eval()

    gen_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    loader = DataLoader(dataset=DualImageFolderDataset("../../datasets/Sketchy/original/all_images",
                                                       "../../datasets/Sketchy/corrupted/all_images",
                                                       transform=gen_transform),
                        batch_size=16,
                        shuffle=True,
                        num_workers=10)

    gan_writer = SummaryWriter(f"eval/logs/train")

    all_images = []

    with tqdm(loader, desc='Train', bar_format=BAR_FORMAT) as tqdm_loader:
        for i, (original, corrupted, _) in enumerate(tqdm_loader):
            original, corrupted = original.to(device), corrupted.to(device)

            with torch.no_grad():
                generated = gen(corrupted)

                batch_images = [generated[0].detach().cpu(), corrupted[0].cpu(), original[0].cpu()]
                all_images.append(torch.stack(batch_images))

            if i == 499:
                break

    all_images_tensor = torch.cat(all_images, dim=0)

    grid = torchvision.utils.make_grid(
        all_images_tensor,
        nrow=3,
        normalize=True
    )

    gan_writer.add_image(tag="{=} Generated <==> Corrupted <==> Original {=}",
                         img_tensor=grid,
                         global_step=0)
    gan_writer.flush()
