import cv2
import random
import numpy as np
from pathlib import Path


def calculate_corruption_percentage(original_image, corrupted_image):
    total_black_pixels = np.sum(original_image < 255)
    remaining_black_pixels = np.sum(corrupted_image < 255)
    removed_black_pixels = total_black_pixels - remaining_black_pixels
    corruption_percentage = (removed_black_pixels / total_black_pixels) * 100
    return corruption_percentage


def generate_corrupted_sketch(image_path, save_path, boundaries):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False

    height, width = image.shape
    while True:
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


def create_corrupted_dataset(original_dir, corrupted_dir, boundaries):
    original_path = Path(original_dir)
    corrupted_path = Path(corrupted_dir)
    corrupted_path.mkdir(parents=True, exist_ok=True)

    for class_folder in original_path.iterdir():
        if class_folder.is_dir():
            print(class_folder)

            class_corrupted_path = corrupted_path / class_folder.name
            class_corrupted_path.mkdir(parents=True, exist_ok=True)

            for image_file in class_folder.iterdir():
                if image_file.is_file() and image_file.suffix in ['.png', '.jpg', '.jpeg']:
                    save_path = class_corrupted_path / image_file.name
                    generate_corrupted_sketch(str(image_file), str(save_path), boundaries)


if __name__ == '__main__':
    original_dir = '../datasets/Sketchy/original/all_images'
    corrupted_dir = '../datasets/Sketchy/corrupted/all_images'
    corruption_percent = (10, 40)

    create_corrupted_dataset(original_dir, corrupted_dir, boundaries)
