import logging

from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from .PyTorch_CIFAR10.cifar10_models.resnet import resnet18


class ResNet18_Client(nn.Module):
    def __init__(self, pretained, num_classes):
        super().__init__()

        # Construct base resnet (pretrained)
        if pretained == True:
            # if num_classes == 10:
            #     model = models.resnet18(pretained=True)
            #     model_urls = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
            #     model.load_state_dict(model_zoo.load_url(model_urls,model_dir='./'))
            #     num_ftrs = model.fc.in_features
            #     model.fc = nn.Linear(num_ftrs, 100)
            # elif num_classes == 100:
            #     self.base = resnet18(pretrained=pretained)
            # else:
            #     logging.info("Unsupport num_classes")
            self.base = resnet18(pretrained=pretained)
        else:
            self.base = models.resnet18(pretrained=False)

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
