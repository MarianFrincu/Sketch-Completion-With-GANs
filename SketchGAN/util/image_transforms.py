import random
import torchvision.transforms.functional as functional


def invert(tensor):
    tensor = 1. - tensor
    return tensor


def random_shift(image, max_shift=32):
    horizontal_shift = random.randint(-max_shift, max_shift)
    vertical_shift = random.randint(-max_shift, max_shift)

    return functional.affine(image, angle=0, translate=[horizontal_shift, vertical_shift], scale=1, shear=[0, 0], fill=[255])
