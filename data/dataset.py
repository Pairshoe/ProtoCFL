import logging
from PIL import Image

import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Cifar_truncated(data.Dataset):
    def __init__(self, args, train=True, data_idxs=None):
        super().__init__()
        self.args = args
        self.data_idxs = data_idxs
        self.loader = pil_loader

        # transform
        if args.dataset_name == 'cifar10':
            CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
            CIFAR_STD = [0.2023, 0.1994, 0.2010]
        elif args.dataset_name == 'cifar100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]
        else:
            logging.error('Unsupport Dataset')
            assert False
        
        if train == True:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ])
        
        # dataset
        if args.dataset_name == 'cifar10':
            self.all_data = CIFAR10(
                root=args.dataset_dir, train=train, transform=self.transform, download=False)
        elif args.dataset_name == 'cifar100':
            self.all_data = CIFAR100(
                root=args.dataset_dir, train=train, transform=self.transform, download=False)
        else:
            logging.error('Unsupport Dataset')
            assert False
        
        self.data, self.labels = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        data = []
        labels = []
        # total is 60000 (train: 50000; test: 10000)
        data_idxs = range(len(self.all_data)) if self.data_idxs is None else self.data_idxs
        for idx in data_idxs:
            data.append(self.all_data[idx][0])
            labels.append(self.all_data[idx][1])

        return data, labels

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        
        return data, label

    def __len__(self):
        return len(self.data)
