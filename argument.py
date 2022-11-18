import os
import argparse


def add_args():
    parser = argparse.ArgumentParser(description='ProtoCFL')

    # general settings
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    # dataset settings
    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--dataset_dir', type=str, default='~/datasets/')
    parser.add_argument('--train_data_idxs', type=list, default=range(5000))
    parser.add_argument('--test_data_idxs', type=list, default=range(1000))
    parser.add_argument('--partition_method', type=str, default='tasks')

    # training settings
    parser.add_argument('--train_batchsize', type=int, default=16)
    parser.add_argument('--client_optimizer', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--epochs_on_client', type=int, default=5)
    parser.add_argument('--epochs_on_server', type=int, default=500)

    # test settings
    parser.add_argument('--test_batchsize', type=int, default=16)

    # experiment setting
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--num_clients', type=int, default=12)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--replay', type=bool, default=False)
    parser.add_argument('--coreset', type=bool, default=False)
    parser.add_argument('--argument', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results/')

    args, unknown = parser.parse_known_args()
    return args


class Arguments:

    def __init__(self, cmd_args):
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            setattr(self, arg_key, arg_val)


def load_arguments():
    cmd_args = add_args()
    args = Arguments(cmd_args)

    if hasattr(args, "dataset_dir"):
        args.dataset_dir = os.path.expanduser(args.dataset_dir)

    return args