from torch import nn
from torch.nn import functional as F


class ResNet18_Server(nn.Module):
    def __init__(self, num_classes = 10, dropout = 0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x


def create_resnet18_server(num_classes):
    return ResNet18_Server(num_classes)
