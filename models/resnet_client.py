import logging

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

from .PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class ResNet18_Client(nn.Module):
    def __init__(self, pretained, num_classes):
        super().__init__()

        # Construct base resnet (pretrained)
        if pretained == True:
            logging.info("Pretrained")
            if num_classes == 10:
                self.base = resnet18(pretrained=pretained)
            elif num_classes == 100:
                self.base = torch.load('/home/pairshoe/ProtoCFL/models/pre-encoder-cifar100.ckpt')
            else:
                logging.info("Unsupport num_classes")
        else:
            logging.info("No Pretrained")
            self.base = models.resnet18()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break;
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x


def create_resnet18_client(pretained, num_classes):
    return ResNet18_Client(pretained, num_classes)
