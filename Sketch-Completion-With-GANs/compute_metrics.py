from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.sketchgan_generator import Generator
from util.text_format_consts import BAR_FORMAT
from util.dual_image_folder_dataset import DualImageFolderDataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = ""

    gen = Generator(in_channels=1, out_channels=1).to(device)
    gan_save = torch.load(f'{model_dir}/save.pth', map_location=device)
    gen.load_state_dict(gan_save['generator_state_dict'])

    gan_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    corrupted_root = Path('test/corrupted')
    image_paths = sorted(list(corrupted_root.rglob("*.png")))

    loader = DataLoader(
        dataset=DualImageFolderDataset(
            first_root='test/original',
            second_root='test/corrupted',
            transform=gan_transform),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    output_dir = Path(model_dir, "generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    gen.eval()

    all_preds = []
    all_targets = []

    batch_start = 0

    for original, corrupted, labels in tqdm(loader, desc='Generating', bar_format=BAR_FORMAT):
        original = original.to(device)
        corrupted = corrupted.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            generated = gen(corrupted)

        is_normalized = (
                original.min().item() < 0 or
                generated.min().item() < 0 or
                corrupted.min().item() < 0
        )

        if is_normalized:
            original = original.mul(0.5).add(0.5).clamp(0, 1)
            generated = generated.mul(0.5).add(0.5).clamp(0, 1)
            corrupted = corrupted.mul(0.5).add(0.5).clamp(0, 1)

        mask_gt = (torch.abs(original - corrupted) > 1e-3).int()
        mask_pred = (torch.abs(generated - corrupted) > 1e-3).int()
        active_mask = ((mask_gt + mask_pred) > 0)

        y_true = mask_gt[active_mask]
        y_pred = mask_pred[active_mask]

        all_targets.append(y_true.cpu())
        all_preds.append(y_pred.cpu())

        batch_paths = image_paths[batch_start:batch_start + len(corrupted)]
        batch_start += len(corrupted)

        for img, path in zip(generated, batch_paths):
            relative_path = path.relative_to(corrupted_root)
            save_path = output_dir / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(img, save_path)

    y_pred = torch.cat(all_preds, dim=0).numpy().ravel()
    y_true = torch.cat(all_targets, dim=0).numpy().ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    print("\n Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
