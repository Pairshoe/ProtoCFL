from torch import nn
from torch.nn import functional as F


class ResNet18_Server(nn.Module):
    def __init__(self, num_classes = 10, dropout = 0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_resnet18_server(num_classes):
    return ResNet18_Server(num_classes)
