import torch
import torch.nn as nn

from torchvision import models

class MRNet_v2(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(MRNet_v2, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.forward(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x