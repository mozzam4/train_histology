from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch
import torch.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from models.PatchResnet import PatchResnet
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.WSIResnet import WSIResnet
from types import SimpleNamespace
from torchcontrib.optim import SWA
model_dict = {}
model_dict['PatchModel'] = WSIResnet()
model_dict['FullImageModel'] = WSIResnet(pretrained=False)


def create_model(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

# define the LightningModule


class LitResnet(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        self.model = create_model(model_name)
        self.loss_module = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['image']
        y = batch['if_msi']
        x = self(x)
        loss = self.loss_module(x, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['if_msi']
        x = self(x)
        loss = self.loss_module(x, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['if_msi']
        x = self(x)
        x = torch.nn.Sigmoid()(x)
        loss = self.loss_module(x, y)
        #y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == x).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        #opt = SWA(optimizer, swa_start=5, swa_freq=5, swa_lr=0.005)
        # scheduler = CosineAnnealingLR(optimizer, T_max=100)
        # swa_start = 5
        # swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        return opt

