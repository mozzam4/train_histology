from torchvision.models import resnet50, ResNet50_Weights
import torch


class PatchResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fc1 = torch.nn.Linear(1000, 256)
        self.fc2 = torch.nn.Linear(256, 56)
        self.fc3 = torch.nn.Linear(56, 1)
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = torch.nn.functional.elu_(x)
        x = self.fc2(x)
        x = torch.nn.functional.elu_(x)
        x = self.fc3(x)
        #x = self.softmax(x)
        x = self.sigmoid(x)
        return x
