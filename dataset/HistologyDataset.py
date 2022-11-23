import os
import io
import PIL.Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class HistologyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.annotation = pd.read_csv(csv_file, encoding='utf-8')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        total = 0
        for ind, r in self.annotation.iterrows():
            total = total + int(r['no_of_files'])
        return total

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        total = 0
        itm_index = index
        # Default Value
        patch_index = itm_index
        img_name = self.annotation.iloc[0]['name']
        if_msi = int(self.annotation.iloc[0]['if_msi'])

        # getting real values
        for ind, r in self.annotation.iterrows():
            total = total + int(r['no_of_files'])
            if index < total:
                patch_index = itm_index
                img_name = r['name']
                if_msi = r['if_msi']
                break
            else:
                itm_index = itm_index - int(r['no_of_files']) + 1

        img_name = os.path.join(self.root_dir,
                                img_name, str(patch_index))
        image = PIL.Image.open(img_name + '.jpg')
        if_msi = torch.tensor(np.array([if_msi]).astype(np.float32))
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'if_msi': if_msi}
        return sample

