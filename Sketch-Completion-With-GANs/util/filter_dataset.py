import json
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def read_invalid_names(invalid_files):
    names = set()
    for invalid_file in invalid_files:
        with open(invalid_file) as file:
            names.update(Path(line.strip()).stem for line in file if line.strip())
    return names


def filter_dataset(sketch_dir, output_dir, invalid_files):
    Path(output_dir).mkdir(parents=True)

    invalid_names = read_invalid_names(invalid_files)
    kept, removed = 0, 0

    for class_folder in sorted(Path(sketch_dir).iterdir()):
        if not class_folder.is_dir():
            continue

        output_class_folder = Path(output_dir, class_folder.name)
        output_class_folder.mkdir(parents=True, exist_ok=True)

        for image_file in sorted(class_folder.iterdir()):
            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if image_file.stem in invalid_names:
                removed += 1
            else:
                shutil.copy2(image_file, output_class_folder / image_file.name)
                kept += 1

    print(f"Kept {kept} sketches, removed {removed}")


if __name__ == '__main__':
    current_dir = Path(__file__).parent

    with open(Path(current_dir, "config.json"), 'r') as file:
        config = json.load(file)['filter_dataset']

    filter_dataset(sketch_dir=Path(current_dir, config['sketch_dir']),
                   output_dir=Path(current_dir, config['output_dir']),
                   invalid_files=[Path(current_dir, path) for path in config['invalid_files']])
