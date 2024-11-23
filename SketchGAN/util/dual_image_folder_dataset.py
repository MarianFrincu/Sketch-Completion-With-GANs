from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class DualImageFolderDataset(Dataset):
    def __init__(self, first_root, second_root, transform=None, target_transform=None):
        self.first_dataset = ImageFolder(first_root, transform=transform, target_transform=target_transform)
        self.second_dataset = ImageFolder(second_root, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.first_dataset)

    def __getitem__(self, index):
        first_img, first_label = self.first_dataset[index]
        second_img, second_label = self.second_dataset[index]

        assert first_label == second_label, "The images have different label"

        return first_img, second_img, first_label
