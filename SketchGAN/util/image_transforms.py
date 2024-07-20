def binarize(tensor, threshold=0.5):
    tensor[tensor < threshold] = 0.
    return tensor


def invert(tensor):
    tensor = 1. - tensor
    return tensor
