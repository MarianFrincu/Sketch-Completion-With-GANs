
def binarize(tensor):
    tensor[tensor < 1.] = 0.
    tensor = 1. - tensor
    return tensor
