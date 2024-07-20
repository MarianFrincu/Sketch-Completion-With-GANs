def binarize(tensor, threshold=0.5):
    tensor[tensor < threshold] = 0.
    tensor = 1. - tensor
    return tensor
