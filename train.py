import pytorch_lightning as pl
from LighteningHistology import LitResnet
from dataset.HistologyDataset import HistologyDataset
from dataset.HistoFullImageDataset import HistoFullImageDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize, CenterCrop, Compose
import torch.utils.data as data
import torch
from LighteningDataHistology import LighteningDataHistology
import argparse
import sys
import os
from pathlib import Path


def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('inputDirectory', help='Path to the input directory.')
    parser.add_argument('selectModel', help='1 for Full image  based, 0 for patches image')
    parser.add_argument('maxEpochs', help='Max number of Epochs')
    return parser


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if not os.path.exists(parsed_args.inputDirectory):
        print("Input file path does not exist")

    if parsed_args.selectModel == str(1):
        csv_file = os.path.join(parsed_args.inputDirectory, Path('Few_patches/annotations_full.csv'))
        root_dir = os.path.join(parsed_args.inputDirectory, Path('Few_patches/selected_images_cropped'))
        Ldm = LighteningDataHistology(csv_file, root_dir, int(parsed_args.selectModel), batch_size=1, if_pretrained=False)
        resnet = LitResnet(model_name='FullImageModel')
    else:
        csv_file = os.path.join(parsed_args.inputDirectory, Path('Few_patches/annotations.csv'))
        root_dir = os.path.join(parsed_args.inputDirectory, Path('Few_patches/sel'))
        Ldm = LighteningDataHistology(csv_file, root_dir, int(parsed_args.selectModel), batch_size=256, if_pretrained=True)
        resnet = LitResnet(model_name='PatchModel')
#
# batch_num = 1
# if training_patch is False:
#     # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
#     train_dataset = HistoFullImageDataset(csv_file=r'/home/mozzam/Documents/Few_patches/annotations_full.csv',
#                                           root_dir=r'/home/mozzam/Documents/Few_patches/selected_images_cropped',
#                                           transform=Compose([Resize(1024),
#                                                              CenterCrop(1024),
#                                                              ToTensor(),
#                                                              Normalize(mean=[0.485, 0.456, 0.406],
#                                                                        std=[0.229, 0.224, 0.225])]))
#
# else:
#     # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
#     train_dataset = HistologyDataset(csv_file=r'/home/mozzam/Documents/Few_patches/annotations.csv',
#                                      root_dir=r'/home/mozzam/Documents/Few_patches/patches',
#                                      transform=Compose([Resize(256),
#                                                         CenterCrop(224),
#                                                         ToTensor(),
#                                                         Normalize(mean=[0.485, 0.456, 0.406],
#                                                                   std=[0.229, 0.224, 0.225])]))
#     batch_num = 4
#
# train_set_size = int(len(train_dataset) * 0.7)
# test_set_size = int(max(len(train_dataset) * 0.1, 1))
# valid_set_size = len(train_dataset) - train_set_size - test_set_size
# train_set, valid_set, test_set = data.random_split(train_dataset, [train_set_size, valid_set_size, test_set_size])
#
#
# train_dataloader = DataLoader(train_set, batch_size=batch_num,
#                               shuffle=True, num_workers=0)
#
# val_dataloader = DataLoader(valid_set, batch_size=batch_num,
#                             shuffle=True, num_workers=0)
#
# test_dataloader = DataLoader(test_set, batch_size=batch_num,
#                              shuffle=True, num_workers=0)

    trainer = pl.Trainer(max_epochs=parsed_args.maxEpochs, gpus=4, num_nodes=1, strategy='ddp')
    # trainer = pl.Trainer(max_epochs=1, gpus=0)
    trainer.fit(model=resnet, datamodule=Ldm)
    trainer.test(datamodule=Ldm)

