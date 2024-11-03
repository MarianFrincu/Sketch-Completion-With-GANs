import torch
import random
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms


def invert(tensor):
    tensor = 1. - tensor
    return tensor


def random_shift(image, max_shift=32):
    horizontal_shift = random.randint(-max_shift, max_shift)
    vertical_shift = random.randint(-max_shift, max_shift)

    return functional.affine(image, angle=0, translate=[horizontal_shift, vertical_shift], scale=1, shear=[0, 0], fill=[255])


def crop_detected_region(original, corrupted, image_to_crop):
    batch_size, channels, height, width = original.size()

    generated_regions = []

    for i in range(batch_size):
        difference = torch.abs(original[i] - corrupted[i])

        mask = difference.sum(dim=0) > 0

        if mask.sum() == 0:
            generated_regions.append(transforms.CenterCrop((128, 128))(corrupted[i]))
            continue

        y_coords, x_coords = torch.where(mask)

        min_x, max_x = x_coords.min().item(), x_coords.max().item()
        min_y, max_y = y_coords.min().item(), y_coords.max().item()

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        start_x = max(center_x - 64, 0)
        start_y = max(center_y - 64, 0)

        end_x = min(start_x + 128, width)
        end_y = min(start_y + 128, height)

        if end_x - start_x < 128:
            start_x = max(end_x - 128, 0)
        if end_y - start_y < 128:
            start_y = max(end_y - 128, 0)

        generated_region = image_to_crop[i][:, start_y:end_y, start_x:end_x]

        generated_regions.append(generated_region)

    return torch.stack(generated_regions)


