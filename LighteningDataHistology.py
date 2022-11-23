from typing import Optional
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from dataset.HistologyDataset import HistologyDataset
from dataset.HistoFullImageDataset import HistoFullImageDataset
from torchvision.transforms import ToTensor, Resize, Normalize, CenterCrop, Compose


class LighteningDataHistology(pl.LightningDataModule):

    def __init__(self, annotation_dir, root_dir, if_patch=True, batch_size=4, if_pretrained=True):
        super().__init__()
        self.batch_size = batch_size
        self.if_patch = if_patch
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.test_set = None
        self.valid_set = None
        self.train_set = None

        if if_pretrained is True:
            # size required for pretrained model
            self.rescale_size = 256
            self.centerCrop = 224
        else:
            self.rescale_size = 1024
            self.centerCrop = 1024

    def setup(self, stage: Optional[str] = None):

        if self.if_patch is False:
            train_dataset = HistoFullImageDataset(csv_file=self.annotation_dir,
                                                  root_dir=self.root_dir,
                                                  transform=Compose([Resize(self.rescale_size),
                                                                     CenterCrop(self.centerCrop),
                                                                     ToTensor(),
                                                                     Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])]))

        else:
            train_dataset = HistologyDataset(csv_file=self.annotation_dir,
                                             root_dir=self.root_dir,
                                             transform=Compose([Resize(self.rescale_size),
                                                                CenterCrop(self.centerCrop),
                                                                ToTensor(),
                                                                Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]))
        train_set_size = int(len(train_dataset) * 0.7)
        test_set_size = int(max(len(train_dataset) * 0.1, 1))
        valid_set_size = len(train_dataset) - train_set_size - test_set_size
        self.train_set, self.valid_set, self.test_set = data.random_split(train_dataset, [train_set_size,
                                                                                          valid_set_size,
                                                                                          test_set_size])

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=0)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=0)
