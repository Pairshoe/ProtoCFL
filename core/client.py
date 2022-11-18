import torch

class Client():
    def __init__(self, args, device, model):
        self.args = args
        self.device = device
        self.model = model
        self.id = 0

    def set_id(self, id):
        self.id = id

    def train(self, train_data):
        pass

    def inference(self, train_data):
        # Only perform validation not training,
        # No need to calculate gradients for forward and backward phase.
        with torch.no_grad():
            model = self.model

            model.to(self.device)
            model.eval()

            features = []
            for (x, labels) in train_data:
                x, labels = x.to(self.device), labels.to(self.device)
                feat = model(x)
                for id in range(len(labels)):
                    features.append([feat[id].cpu(), labels[id].cpu()])
        
        return features
