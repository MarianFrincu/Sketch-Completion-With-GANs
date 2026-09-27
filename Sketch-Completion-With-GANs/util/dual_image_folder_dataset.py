from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class DualImageFolderDataset(Dataset):
    def __init__(self, first_root, second_root, transform=None, target_transform=None):
        self.first_dataset = ImageFolder(first_root, transform=transform, target_transform=target_transform)
        self.second_dataset = ImageFolder(second_root, transform=transform, target_transform=target_transform)

        if relative_paths(self.first_dataset) != relative_paths(self.second_dataset):
            raise ValueError(f"{first_root} and {second_root} do not contain the same images")

    def __len__(self):
        return len(self.first_dataset)

    def __getitem__(self, index):
        first_img, label = self.first_dataset[index]
        second_img, _ = self.second_dataset[index]

        return first_img, second_img, label


def relative_paths(dataset):
    return [Path(path).relative_to(dataset.root) for path, _ in dataset.samples]
