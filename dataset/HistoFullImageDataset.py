import os
import io
import PIL.Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class HistoFullImageDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotation = pd.read_csv(csv_file, encoding='utf-8')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        if_msi = self.annotation.iloc[index, 1]
        image = PIL.Image.open(img_name)
        if_msi = torch.tensor(np.array([if_msi]).astype(np.float32))
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'if_msi': if_msi}
        return sample



