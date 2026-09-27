import json
import math
import random
import shutil
from pathlib import Path


def split_dataset(original_dir, corrupted_dir, split_dir, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2, seed=42):
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("The sum of ratios must be 1.")

    Path(split_dir).mkdir(parents=True)

    rng = random.Random(seed)

    for class_folder in sorted(Path(original_dir).iterdir()):
        if not class_folder.is_dir():
            continue

        corrupted_class_folder = Path(corrupted_dir, class_folder.name)
        images = sorted(image.name for image in class_folder.iterdir()
                        if image.is_file() and (corrupted_class_folder / image.name).is_file())
        rng.shuffle(images)

        train_end = round(len(images) * train_ratio)
        val_end = round(len(images) * (train_ratio + val_ratio))
        subsets = {'train': images[:train_end], 'val': images[train_end:val_end], 'test': images[val_end:]}

        for subset, subset_images in subsets.items():
            if not subset_images:
                continue

            for kind, source_folder in (('original', class_folder), ('corrupted', corrupted_class_folder)):
                destination = Path(split_dir, subset, kind, class_folder.name)
                destination.mkdir(parents=True, exist_ok=True)
                for image_name in subset_images:
                    shutil.copy2(source_folder / image_name, destination / image_name)


if __name__ == '__main__':
    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['split_dataset']

    split_dataset(original_dir=Path(current_dir, config['original_dir']),
                  corrupted_dir=Path(current_dir, config['corrupted_dir']),
                  split_dir=Path(current_dir, config['split_dir']),
                  train_ratio=config['train_ratio'],
                  val_ratio=config['val_ratio'],
                  test_ratio=config['test_ratio'],
                  seed=config['seed'])
