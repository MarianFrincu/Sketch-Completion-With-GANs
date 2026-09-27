import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from models.sketchgan_generator import load_generator
from util.dual_image_folder_dataset import DualImageFolderDataset, relative_paths
from util.postprocess import apply_postprocess
from util.text_format_consts import BAR_FORMAT

if __name__ == "__main__":
    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['compute_metrics']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = load_generator(Path(current_dir, config['checkpoint']), device)
    generator.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = DualImageFolderDataset(
        first_root=Path(current_dir, config['data_dir'], 'original'),
        second_root=Path(current_dir, config['data_dir'], 'corrupted'),
        transform=transform)

    image_paths = relative_paths(dataset.second_dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    batch_start = 0

    for original, corrupted, _ in tqdm(loader, desc='Generating', bar_format=BAR_FORMAT):
        original = original.to(device)
        corrupted = corrupted.to(device)

        with torch.no_grad():
            generated = generator.denormalize(generator(generator.normalize(corrupted))).clamp(0, 1)

        if config['postprocess']:
            generated = torch.stack([apply_postprocess(image) for image in generated]).to(device)

        mask_gt = torch.abs(original - corrupted) > 1e-3
        mask_pred = torch.abs(generated - corrupted) > 1e-3

        true_positives += (mask_gt & mask_pred).sum().item()
        false_positives += (~mask_gt & mask_pred).sum().item()
        false_negatives += (mask_gt & ~mask_pred).sum().item()

        batch_start += len(generated)

    active_pixels = true_positives + false_positives + false_negatives

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * true_positives / (active_pixels + true_positives) if active_pixels else 0.0
    accuracy = true_positives / active_pixels if active_pixels else 0.0

    print("\n Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
