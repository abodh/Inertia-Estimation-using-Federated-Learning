import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return data, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.selected_clients = []

        # splitting the dataset and setting the batch size for the local training
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, epocc, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=0.0005)

        epoch_loss = []
        for iter in range(self.args.local_ep): # train for local epoch number of times
            batch_loss = []
            for batch_idx, (data, labels) in enumerate(self.ldr_train):
                data, labels = data.to(self.args.device), labels.to(self.args.device)
                if self.args.model == 'cnn':
                    data = data.double().unsqueeze(1)
                    labels = labels.double().view(-1, 1)
                else:
                    data = data.float()
                    labels = labels.float().view(-1, 1)
                # getting the log probability of output -> softmax output
                # log_probs = net(images)

                net.zero_grad()
                optimizer.zero_grad()
                oupt = net(data)
                # if (epocc > 0) and (epocc % 150 == 0):
                #     pdb.set_trace()

                '''
                
                for eg.
                for 4 output class
                if o/p -> [0.1 0.2 0.19 0.9] this is 3
                the label will have -> [0 0 0 1]
                based on this, the loss will be calculated
                
                '''

                loss = self.loss_func(oupt, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
