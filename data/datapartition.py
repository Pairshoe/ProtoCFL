import logging
import random
from collections import defaultdict


def partition(args, train_dataset, test_dataset):
    # classify
    train_dataset_dict = defaultdict(list)
    for i in range(len(train_dataset)):
        label = train_dataset.labels[i]
        train_dataset_dict[label].append(train_dataset[i])

    test_dataset_dict = defaultdict(list)
    for i in range(len(test_dataset)):
        label = test_dataset.labels[i]
        test_dataset_dict[label].append(test_dataset[i])
    
    # partition
    # dataset classes: 0 1 2 3 4 5 6 7 8 9
    # task_0: 0 1 2 3 4 5
    # - client_0: 0 1, client_1: 1 2, ..., client_4: 5 0, client_5: 0 1, client_6: 1 2, ..., client_11: 5 0
    # task_1: 4 5 6 7 8 9
    # - client_0: 4 5, client_1: 5 6, ..., client_4: 9 4, client_5: 4 5, client_6: 5 6, ..., client_11: 9 4
    num_tasks = args.num_tasks
    num_clients = args.num_clients
    logging.debug(f"num_tasks: {num_tasks}.")
    logging.debug(f"num_clients: {num_clients}.")

    overlap_ratio = 0.1

    num_classes = len(train_dataset_dict)
    num_classes_overlap_for_task = int(num_classes * overlap_ratio)
    num_classes_for_task = num_classes // num_tasks + num_classes_overlap_for_task
    num_classes_disjoint_for_task = num_classes // num_tasks - num_classes_overlap_for_task
    logging.debug(f"num_classes: {num_classes}.")
    logging.debug(f"num_classes_for_task: {num_classes}.")
    logging.debug(f"num_classes_disjoint_for_task: {num_classes_disjoint_for_task}.")
    logging.debug(f"num_classes_overlap_for_task: {num_classes_overlap_for_task}.")

    num_classes_for_client = int(0.2 * num_classes)
    num_classes_disjoint_for_client = int(0.1 * num_classes)
    logging.debug(f"num_classes_for_client: {num_classes_for_client}.")
    logging.debug(f"num_classes_disjoint_for_client: {num_classes_disjoint_for_client}.")
    
    train_dataset_global = {}
    test_dataset_global = {}
    train_dataset_local_dict = {}
    test_dataset_local_dict = {}
    labels = random.sample(range(num_classes), num_classes)

    for task_id in range(args.num_tasks):
        # global
        train_dataset_global[task_id] = []
        test_dataset_global[task_id] = []
        for label_id in range(
            task_id * num_classes_disjoint_for_task, \
            (task_id + 1) * num_classes_disjoint_for_task + num_classes_overlap_for_task * 2):
            train_dataset_global[task_id] += train_dataset_dict[labels[label_id]]
            test_dataset_global[task_id] += test_dataset_dict[labels[label_id]]
        # local
        train_dataset_local_dict[task_id] = defaultdict(list)
        test_dataset_local_dict[task_id] = defaultdict(list)
        for client_id in range(args.num_clients):
            for label_id in range(
                client_id * num_classes_disjoint_for_client, \
                client_id * num_classes_disjoint_for_client + num_classes_for_client):
                idx = task_id * num_classes_disjoint_for_task + label_id % num_classes_for_task
                train_dataset_local_dict[task_id][client_id] += train_dataset_dict[labels[idx]]
                test_dataset_local_dict[task_id][client_id] += test_dataset_dict[labels[idx]]
    
    return train_dataset_global, test_dataset_global, train_dataset_local_dict, test_dataset_local_dict
