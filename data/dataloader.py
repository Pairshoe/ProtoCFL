import logging

from torch.utils.data import DataLoader

from .dataset import Cifar_truncated
from .datapartition import partition


def load_data(args):
    # dataset
    logging.info(f'Dataset: {args.dataset_name}.')
    train_dataset = Cifar_truncated(args, train=True, data_idxs= args.train_data_idxs)
    test_dataset = Cifar_truncated(args, train=False, data_idxs= args.test_data_idxs)

    # partition
    logging.info(f'Data Partition.')
    train_data_global, test_data_global, \
    train_data_local_dict, test_data_local_dict = partition(args, train_dataset, test_dataset)

    # dataloader
    logging.info(f'Data Loader.')
    for task_id in range(args.num_tasks):
        logging.debug(f"task {task_id} global: {len(train_data_global[task_id])}")
        logging.debug(f"task {task_id} global: {len(test_data_global[task_id])}")
        train_data_global[task_id] = DataLoader(
            dataset= train_data_global[task_id], batch_size=args.train_batchsize, \
            shuffle=True, drop_last=True)
        test_data_global[task_id] = DataLoader(
            dataset= test_data_global[task_id], batch_size=args.test_batchsize, \
            shuffle=True, drop_last=True)
        for client_id in range(args.num_clients):
            logging.debug(f"task {task_id} client {client_id}: {len(train_data_local_dict[task_id][client_id])}")
            logging.debug(f"task {task_id} client {client_id}: {len(test_data_local_dict[task_id][client_id])}")
            train_data_local_dict[task_id][client_id] = DataLoader(
                dataset= train_data_local_dict[task_id][client_id], batch_size=args.train_batchsize, \
                shuffle=True, drop_last=True)
            test_data_local_dict[task_id][client_id] = DataLoader(
                dataset= test_data_local_dict[task_id][client_id], batch_size=args.test_batchsize, \
                shuffle=True, drop_last=True)

    return train_data_global, test_data_global, train_data_local_dict, test_data_local_dict
