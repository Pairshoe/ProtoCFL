import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

from argument import load_arguments
from data import Cifar_truncated

def init():
    args = load_arguments()

    logging.basicConfig(
        # filename=args.results_dir + 'log.txt',
        format='[%(levelname)s]%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s',
        datefmt='%m-%d %H:%M:%S', level=logging.DEBUG)
    logging.addLevelName(logging.INFO, "\033[31;21m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.DEBUG, "\033[33;21m%s\033[1;0m" % logging.getLevelName(logging.DEBUG))
    logging.getLogger('matplotlib.font_manager').disabled = True

    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return args


if __name__ == "__main__":
    # init framework
    args = init()
    args.dataset_name='cifar100'

    # init device
    device = torch.device("cuda:" + str(args.gpu) \
        if torch.cuda.is_available() else "cpu")

    # load data
    # transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # trainset = torchvision.datasets.CIFAR100('../datasets', train=True, download=False, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batchsize, shuffle=True, num_workers=8)

    # t = transforms.Compose([transforms.ToTensor()])
    # testset = torchvision.datasets.CIFAR100(root='../datasets', train=False,download=False, transform=t)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batchsize,shuffle=False, num_workers=8)
    trainset = Cifar_truncated(args, train=True, data_idxs= args.train_data_idxs)
    testset = Cifar_truncated(args, train=False, data_idxs= args.test_data_idxs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batchsize, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.train_batchsize, shuffle=True, num_workers=8)

    # model = models.resnet18(pretrained=True)

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 100)
    model = torch.load('HW4_II_para.ckpt')
    model.cuda()

    classifier = nn.Linear(512, 100)
    classifier.cuda()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
    
        total_right = 0
        total = 0
        
        for data in trainloader:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(),Variable(labels).cuda()
            
            optimizer.zero_grad()
            
            inputs = F.interpolate(inputs, [224,224])

            for name, module in model._modules.items():
                if name == 'avgpool':
                    break;
                inputs = module(inputs)

            inputs = F.avg_pool2d(inputs, inputs.size()[2:])
            inputs = inputs.view(inputs.size(0), -1)

            outputs = classifier(inputs)

            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            
            predicted = outputs.data.max(1)[1]
            total += labels.size(0)
            total_right += (predicted == labels.data).float().sum()
            
        print("Training accuracy for epoch {} : {}".format(epoch+1,100*total_right/total))
        
        total_right = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images,labels = data
                images, labels = Variable(images).cuda(),Variable(labels).cuda()
                
                images = F.interpolate(images, [224,224])

                for name, module in model._modules.items():
                    if name == 'avgpool':
                        break;
                    images = module(images)

                images = F.avg_pool2d(images, images.size()[2:])
                images = images.view(images.size(0), -1)

                
                outputs = classifier(images)
            
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                total_right += (predicted == labels.data).float().sum()
            
        print("Test accuracy: %d" % (100*total_right/total))
