import logging
import math
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Server():
    def __init__(self, args, device, model, encoder):
        self.args = args
        self.device = device
        self.model = model
        self.encoder = encoder
        self.id = 0
        self.features = []

    def set_features(self, features):
        self.features += features

    def argument_data(self, coreset, scale=3, level=10):
        train_data = []
        # classify
        data_dict = defaultdict(list)
        for core in coreset:
            data_dict[core[1].item()].append(core[0])
        # argument
        for label, data in data_dict.items():
            for _ in range(math.ceil(len(data) * scale)):
                weights = [random.choice(range(level)) for _ in range(len(data))]
                new_data = sum([d * w for d, w in zip(data, weights)]) / sum(weights)
                train_data.append((new_data, label))
        return train_data

    # train on server with features
    def train(self, current_features, replay=False, argument=False):
        model = self.model

        model.to(self.device)
        model.train()

        # replay
        if replay == True:
            train_data_list = self.features
        else:
            train_data_list = current_features
        # argument
        if argument == True:
            train_data_list = self.argument_data(train_data_list, scale=self.args.argument_scale)

        train_data = DataLoader(train_data_list, shuffle=True, batch_size=self.args.train_batchsize)

        # setting
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                amsgrad=True)

        # train
        epoch_loss = []
        for epoch in range(self.args.epochs_on_server):
            batch_loss = []
            total_right = 0
            total = 0

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(x)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                predicted = outputs.data.max(1)[1]
                total += labels.size(0)
                total_right += (predicted == labels.data).float().sum()

                # if batch_idx % 10 == 0:
                #     logging.info(
                #         "Epoch: {}/{} | Batch: {}/{} | Loss: {}".format(
                #             epoch + 1,
                #             args.epochs_on_server,
                #             batch_idx,
                #             len(train_data),
                #             loss.item()
                #         )
                #     )
            logging.debug("Training accuracy for epoch {} : {}".format(epoch, 100 * total_right / total))
                
            if epoch % 50 == 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(Local Training Epoch: {} \tLoss: {:.6f})".format(
                        epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )

    def test(self, test_data):
        model = self.model
        encoder = self.encoder

        model.to(self.device)
        model.eval()
        encoder.to(self.device)
        encoder.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_total": 0,
        }

        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                feat = encoder(x)
                pred = model(feat)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        
        logging.info("Test: Acc = {} / {} = {}".format(metrics["test_correct"], metrics["test_total"], metrics["test_correct"]/metrics["test_total"]))
        logging.info("Test: Loss = {} / {} = {}".format(metrics["test_loss"], metrics["test_total"], metrics["test_loss"]/metrics["test_total"]))
