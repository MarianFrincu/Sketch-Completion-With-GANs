import os
import shutil

from sklearn.model_selection import train_test_split


def split_original_dataset(original_dir, original_split_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of ratios must be 1."

    if not os.path.exists(original_split_dir):
        os.makedirs(original_split_dir)

    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(original_split_dir, subset)
        if not os.path.exists(subset_path):
            os.makedirs(subset_path)

    for class_folder in os.listdir(original_dir):
        class_folder_path = os.path.join(original_dir, class_folder)
        if os.path.isdir(class_folder_path):
            images = [os.path.join(class_folder_path, img) for img in os.listdir(class_folder_path) if
                      os.path.isfile(os.path.join(class_folder_path, img))]

            train_images, test_images = train_test_split(images, test_size=(1 - train_ratio))
            val_images, test_images = train_test_split(test_images, test_size=(test_ratio / (test_ratio + val_ratio)))

            for subset, subset_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
                subset_class_path = os.path.join(original_split_dir, subset, class_folder)
                os.makedirs(subset_class_path, exist_ok=True)
                for image in subset_images:
                    shutil.copy2(image, os.path.join(str(subset_class_path), os.path.basename(image)))


def split_corrupted_dataset(original_split_dir, corrupted_dir, corrupted_split_dir):
    if not os.path.exists(corrupted_split_dir):
        os.makedirs(corrupted_split_dir)

    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(original_split_dir, subset)
        corrupted_subset_path = os.path.join(corrupted_split_dir, subset)

        for class_folder in os.listdir(subset_path):
            original_class_folder_path = os.path.join(subset_path, class_folder)
            corrupted_class_folder_path = os.path.join(corrupted_dir, class_folder)
            corrupted_subset_class_path = os.path.join(corrupted_subset_path, class_folder)

            if os.path.isdir(original_class_folder_path):
                os.makedirs(corrupted_subset_class_path, exist_ok=True)

                for image_name in os.listdir(original_class_folder_path):
                    original_image_path = os.path.join(original_class_folder_path, image_name)
                    corrupted_image_path = os.path.join(corrupted_class_folder_path, image_name)

                    if os.path.isfile(original_image_path) and os.path.isfile(corrupted_image_path):
                        shutil.copy2(corrupted_image_path, os.path.join(corrupted_subset_class_path, image_name))


if __name__ == '__main__':
    original_directory = '../datasets/Sketchy/original/all_images'
    original_split_directory = '../datasets/Sketchy/original'
    corrupted_directory = '../datasets/Sketchy/corrupted/all_images'
    corrupted_split_directory = '../datasets/Sketchy/corrupted'

    split_original_dataset(original_directory,
                           original_split_directory,
                           train_ratio=0.65,
                           val_ratio=0.20,
                           test_ratio=0.15)
    split_corrupted_dataset(original_split_directory,
                            corrupted_directory,
                            corrupted_split_directory)
