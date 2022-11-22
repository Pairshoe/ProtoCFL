import logging
import torch
import numpy as np
from collections import defaultdict

from utils import visualize_features


class Runner:
    def __init__(self, args, device, clients, server):
        self.args = args
        self.device = device
        self.clients = clients
        self.server = server

    def distill(self, features, num_per_label):
        coreset = []
        # classify
        feat_dict = defaultdict(list)
        for feature in features:
            feat_dict[feature[1].item()].append(feature[0])
        # average
        for label, feats in feat_dict.items():
            dists = []
            mean = sum(feats) / len(feats)
            for feat in feats:
                dist = (feat - mean).pow(2).sum(axis=0).sqrt()
                dists.append(dist)
            num = min(num_per_label, len(dists))
            for _ in range(num):
                min_dist = min(dists)
                idx_min_dist = dists.index(min_dist)
                coreset.append((feats[idx_min_dist], torch.tensor(label)))
                del dists[idx_min_dist]
        return coreset

    def run(self, task_id, train_data, test_data_dict, coreset=False):
        # collect this round's features
        logging.info("Collect Features from Clients.")
        c_features = []
        for client in self.clients:
            logging.debug(f"Client {client.id} train_data size: {len(train_data[client.id])} (batchsize = 32).")
            c_feature = client.inference(train_data[client.id])
            logging.debug(f"Client {client.id} c_feature size: {len(c_feature)}.")
            c_features += c_feature
            if self.args.visualize == True:
                visualize_features(c_feature, self.args.results_dir + 'features_' + str(task_id) + '_client_' + str(client.id) + '.png', self.args.random_seed)
        
        # visualize features
        if self.args.visualize == True:
            visualize_features(c_features, self.args.results_dir + 'features_' + str(task_id) + '.png', self.args.random_seed)
        
        # pass this round's features to server
        logging.info("Pass Features to Server.")
        if coreset == True:
            # coreset
            coreset = self.distill(c_features, self.args.num_prototypes)
            visualize_features(coreset, self.args.results_dir + 'coreset_' + str(task_id) + '.png', self.args.random_seed)

            self.server.set_features(coreset)

            # train server
            logging.info("Train on Server.")
            self.server.train(coreset, replay=self.args.replay, argument=self.args.argument)
        else:
            self.server.set_features(c_features)
            
            # train server
            logging.info("Train on Server.")
            self.server.train(c_features, replay=self.args.replay, argument=self.args.argument)

        # test server
        logging.info("Test on Server.")
        for task_id in range(task_id + 1):
            logging.info(f"Test on Task {task_id}.")
            for client_id in range(self.args.num_clients):
                logging.info(f"Test on Client {client_id}.")
                self.server.test(test_data_dict[task_id][client_id])
