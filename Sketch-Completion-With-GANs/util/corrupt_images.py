import json
import cv2
import random
import numpy as np
from pathlib import Path

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def calculate_corruption_percentage(original_image, corrupted_image):
    total_black_pixels = np.sum(original_image < 255)
    remaining_black_pixels = np.sum(corrupted_image < 255)
    removed_black_pixels = total_black_pixels - remaining_black_pixels
    corruption_percentage = (removed_black_pixels / total_black_pixels) * 100
    return corruption_percentage


def generate_corrupted_sketch(image_path, save_path, boundaries, max_attempts=1000):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or np.all(image == 255):
        return False

    height, width = image.shape
    for _ in range(max_attempts):
        mask_height = random.randint(1, height)
        mask_width = random.randint(1, width)
        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        corrupted_image = image.copy()
        corrupted_image[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 255

        corruption_percentage = calculate_corruption_percentage(image, corrupted_image)
        if boundaries[0] <= corruption_percentage <= boundaries[1]:
            cv2.imwrite(save_path, corrupted_image)
            return True

    return False


def create_corrupted_dataset(original_dir, corrupted_dir, boundaries):
    Path(corrupted_dir).mkdir(parents=True)

    failed = []

    for class_folder in sorted(Path(original_dir).iterdir()):
        if not class_folder.is_dir():
            continue

        print(class_folder.name)

        class_corrupted_path = Path(corrupted_dir, class_folder.name)
        class_corrupted_path.mkdir(parents=True, exist_ok=True)

        for image_file in sorted(class_folder.iterdir()):
            if image_file.suffix.lower() in IMAGE_EXTENSIONS:
                save_path = class_corrupted_path / image_file.name
                if not generate_corrupted_sketch(str(image_file), str(save_path), boundaries):
                    failed.append(image_file)

    print(f"Could not corrupt {len(failed)} images, they will be left out of the split.")
    for image_file in failed:
        print(image_file)


if __name__ == '__main__':
    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['corrupt_images']

    random.seed(config['seed'])

    create_corrupted_dataset(original_dir=Path(current_dir, config['original_dir']),
                             corrupted_dir=Path(current_dir, config['corrupted_dir']),
                             boundaries=config['corruption_percent'])
