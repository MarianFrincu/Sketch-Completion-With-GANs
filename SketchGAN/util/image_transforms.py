import random
import torchvision.transforms.functional as functional


def normalize(tensor):
    tensor /= 255.0
    return tensor


def binarize(tensor, threshold=0.5):
    tensor[tensor < threshold] = 0
    tensor[tensor >= threshold] = 1
    return tensor


def invert(tensor):
    tensor = 1. - tensor
    return tensor


def random_shift(image, max_shift=32):
    horizontal_shift = random.randint(-max_shift, max_shift)
    vertical_shift = random.randint(-max_shift, max_shift)

    return functional.affine(image, angle=0, translate=[horizontal_shift, vertical_shift], scale=1, shear=[0, 0])