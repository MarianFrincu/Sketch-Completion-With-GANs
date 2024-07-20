import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class CsvDataset(Dataset):
    def __init__(self, csv_file, corrupted_dir, original_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.corrupted_dir = corrupted_dir
        self.original_dir = original_dir
        self.transform = transform

        class_names = pd.read_csv(csv_file)['class_name'].unique()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        image_name = row['image_name']
        class_name = row['class_name']

        corrupted_image_path = os.path.join(self.corrupted_dir, class_name, image_name)
        original_image_path = os.path.join(self.original_dir, class_name, image_name)

        corrupted_image = cv2.cvtColor(cv2.imread(str(corrupted_image_path)),
                                       cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(cv2.imread(str(original_image_path)),
                                      cv2.COLOR_BGR2RGB)

        if self.transform:
            corrupted_image = self.transform(corrupted_image)
            original_image = self.transform(original_image)

        label = self.class_to_idx[class_name]

        return corrupted_image, original_image, label
