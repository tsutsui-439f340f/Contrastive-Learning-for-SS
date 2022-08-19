import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_encoder = models.vgg16(pretrained=True)
        self.cnn_encoder.avgpool = nn.Identity()
        self.cnn_encoder.classifier=nn.Identity()

    def forward(self, x):
        return self.cnn_encoder(x)