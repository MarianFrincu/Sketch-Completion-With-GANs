import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

from models.sketchgan_generator import load_generator
from util.postprocess import apply_postprocess

if __name__ == "__main__":
    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['complete_sketch']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = load_generator(Path(current_dir, config['checkpoint']), device)
    generator.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    output_dir = Path(current_dir, config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in config['images']:
        image = transform(Image.open(Path(current_dir, image_path)).convert('L')).unsqueeze(0).to(device)

        with torch.no_grad():
            completed = generator.denormalize(generator(generator.normalize(image)))[0]

        if config['postprocess']:
            completed = apply_postprocess(completed)

        save_image(completed, output_dir / Path(image_path).name)
