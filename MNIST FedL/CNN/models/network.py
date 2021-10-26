import torch
# from torch import nn
# import torch.nn.functional as F
import pdb

# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout() #dropout is used inorder to avoid overfitting case
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#         return self.softmax(x)

class Net(torch.nn.Module):
    def __init__(self, n_inp, n_hid1, n_hid2, n_out, dropout_rate,
                 weight_ini, dropout_decision=False):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(n_inp, n_hid1)
        self.hid2 = torch.nn.Linear(n_hid1, n_hid2)
        self.oupt = torch.nn.Linear(n_hid2, n_out)
        self.dropout_decision = dropout_decision
        self.dropout = torch.nn.Dropout(dropout_rate)

    # initializing the weights and biases
        torch.nn.init.xavier_uniform_(self.hid1.weight, gain = weight_ini)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight, gain = weight_ini)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight, gain = weight_ini)
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, X):
        z = torch.tanh(self.hid1(X))
        if (self.dropout_decision):
            z = self.dropout(z)
        z = torch.tanh(self.hid2(z))
        z = self.oupt(z)  # no activation, aka Identity()
        return z