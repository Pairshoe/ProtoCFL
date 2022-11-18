from torch import nn
from torch.nn import functional as F

from .PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class ResNet18_Client(nn.Module):
    def __init__(self, pretained):
        super().__init__()

        # Construct base resnet (pretrained)
        self.base = resnet18(pretrained=pretained)

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break;
            x = module(x)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        return x


def create_resnet18_client(pretained):
    return ResNet18_Client(pretained)
