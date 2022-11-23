from torchvision.models import resnet50, ResNet50_Weights
import torch


class WSIResnet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained is True:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(pretrained=False)
        self.fc1 = torch.nn.Linear(1000, 256)
        self.fc2 = torch.nn.Linear(256, 56)
        self.fc3 = torch.nn.Linear(56, 1)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu_(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu_(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
