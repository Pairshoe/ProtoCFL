import random
import logging

import torch
import numpy as np

from argument import load_arguments
from data import load_data
from models import create_resnet18_client
from models import create_resnet18_server
from core import Client
from core import Server
from core import Runner


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

    # init device
    device = torch.device("cuda:" + str(args.gpu) \
        if torch.cuda.is_available() else "cpu")

    # load data
    logging.info("Load Data.")
    train_data_global, \
    test_data_global, \
    train_data_local_dict, \
    test_data_local_dict = load_data(args)

    # create clients
    logging.info("Create Clients.")
    clients = []
    num_clients = args.num_clients
    for client_id in range(num_clients):
        # create model
        if args.dataset_name == 'cifar10':
            model = create_resnet18_client(pretained=args.pretrained, num_classes=10)
        elif args.dataset_name == 'cifar100':
            model = create_resnet18_client(pretained=args.pretrained, num_classes=100)
        client = Client(args, device, model)
        client.set_id(client_id)
        clients.append(client)

    # create server
    logging.info("Create Server.")
    if args.dataset_name == 'cifar10':
        model = create_resnet18_server(num_classes=10)
        encoder = create_resnet18_client(pretained=args.pretrained, num_classes=10)
    elif args.dataset_name == 'cifar100':
        model = create_resnet18_server(num_classes=100)
        encoder = create_resnet18_client(pretained=args.pretrained, num_classes=100)
    server = Server(args, device, model, encoder)

    # start simulation
    runner = Runner(args, device, clients, server)
    for task_id in range(args.num_tasks):
        logging.info(f"Start Task {task_id}.")
        runner.run(task_id, train_data_local_dict[task_id], test_data_local_dict, coreset=args.coreset)
