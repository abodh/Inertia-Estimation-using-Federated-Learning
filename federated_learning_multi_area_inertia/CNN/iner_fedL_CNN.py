from utils import args_parser, mnist_iid, mnist_non_iid
from models.local_train import LocalUpdate
from models.network import Net, Simple1DCNN
from models.test import test_img

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
import copy
import pdb
# import h5py
from datetime import date, datetime
import os
import time


from torch.utils.data import Dataset, DataLoader

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class freq_data(Dataset):
    # Constructor
    def __init__(self, path):
        file_freq = path + 'freq_norm.csv'
        file_rocof = path + 'rocof_norm.csv'
        freq_data, rocof_data = loading(file_freq, file_rocof)
        self.x, self.y = separate_dataset(freq_data, rocof_data)
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # Return the length
    def __len__(self):
        return self.len

def loading(file_freq, file_rocof):
    # loading total data
    # file_f = h5py.File(file_freq, 'r')
    # file_rocof = h5py.File(file_rocof, 'r')
    # f_var = file_f.get('xyzf')
    # rocof_var = file_rocof.get('xyzr')
    # f_var = np.array(f_var).T
    # rocof_var = np.array(rocof_var).T
    f_var = np.genfromtxt(file_freq, delimiter=",")
    rocof_var = np.genfromtxt(file_rocof, delimiter=",")
    return f_var, rocof_var

def separate_dataset(freq_data, rocof_data):
    '''

    :param freq_data: change of frequency data extracted from the matfile
    :param rocof_data: rocof data extracted from the matfile
    :return: separate training dataset for each of the inputs(frequency, rocof, and p) and an output dataset of inertia

    Note: the data have been normalized already in MATLAB

    '''
    # loads = np.genfromtxt('pulses.csv', delimiter=',')
    # loads = loads.transpose()

    total_dataset = np.hstack((freq_data[:,0:-2],rocof_data[:,0:-2], freq_data[:,-1:]))
    # pdb.set_trace()
    # pdb.set_trace()
    # total_dataset = np.random.permutation(total_dataset)
    # train_num = int(0.8 * len(total_dataset))  # number of data to be trained
    # pdb.set_trace()
    # train_f_rf = total_dataset[0:train_num,:-1]
    # train_M_D = total_dataset[0:train_num,-1]
    # test_f_rf = total_dataset[train_num:len(total_dataset), :-1]
    # test_M_D = total_dataset[train_num:len(total_dataset), -1]
    # pdb.set_trace()
    # return train_f_rf, train_M_D, test_f_rf, test_M_D
    x = total_dataset[:,:-1]
    y = total_dataset[:,-1]
    return x, y

if __name__ == '__main__':

    torch.manual_seed(0); np.random.seed(0)

    frac_train = 0.8

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    data_path = "/gpfs/home/abodh.poudyal/fed_MLP/area12_comb_nonIID/manipulated/"
    data_format = ("_IID_" if args.iid == True else "_non-IID_")

    str(date.today().strftime("%d/%m/%Y"))
    output_path = "./log/" + data_format + str(args.model) + "_" + str(args.device) + "_" + str(date.today().strftime("%b-%d-%Y")) + \
                  str(datetime.now().strftime("-%H.%M.%S-"))
    try:
        os.mkdir(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)

    # load dataset and split users
    if not args.dataset == 'iner':
        # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        dataset = freq_data(data_path)
        train_num = int(frac_train * len(dataset))  # number of data for training
        test_num = len(dataset) - train_num  # number of data for validating

        # splitting into training and validation dataset
        dataset_train, dataset_test = torch.utils.data.random_split(dataset, (train_num, test_num))

        # # load separate training and validating dataset
        # dataset_train = DataLoader(training, shuffle=True)
        # dataset_test = DataLoader(validation, shuffle=False)

        # dataset_train = datasets.MNIST('../data/mnist/', train=True)
        # dataset_test = datasets.MNIST('../data/mnist/', train=False)
        # sample users

        if (args.iid == True):
            print ('IID data selected')
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print('non-IID data selected')
            dict_users = mnist_non_iid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    
    print ("Number of communication: {} Number of local epochs: {}".format(args.epochs, args.local_ep))
    # img_size = dataset_train[0][0].shape

    # # build model
    # if args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')

    n_inp = len(dataset_train[0][0])
    n_hid1 = 25
    n_hid2 = 25
    n_out = 1

    # call your neural network model right here
    if args.model == 'cnn':
        net_glob = Simple1DCNN().double().to(args.device)
    else:
        net_glob = Net(n_inp, n_hid1, n_hid2, n_out, 0.5, 0.05, False).to(args.device)

    # display the network structure
    print(n_inp)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    valid_loss = []
    val_accuracy = []
    weight_ih = []
    weight_ho = []
    min_val_RMSE = 1e5
    min_R_epoch = 1e5

    #cv_loss, cv_acc = [], []
    #val_loss_pre, counter = 0, 0
    # net_best = None
    # best_loss = None
    # val_acc_list, net_list = [], []
    criterion = torch.nn.MSELoss()
    ''' federated averaging algorithm from google's paper '''
    for iter in range(args.epochs): # communication rounds
        w_locals, loss_locals = [], [] # initialize weights
        m = max(int(args.frac * args.num_users), 1) # ensures at least one client is selected
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # random set of m clients
        for idx in idxs_users:

            # calls the local update where the clients are trained for local epoch number of times
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            # get weights and loss from the clients
            w, loss = local.train(iter, net=copy.deepcopy(net_glob).to(args.device))

            # accumulating the local loss and local weights
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # aggregate and update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to the global network
        net_glob.load_state_dict(w_glob)

        # weight_ih.append(np.reshape(net_glob.hid1.weight.data.clone().cpu().numpy(), (1, n_inp * n_hid1)))
        weight_ho.append(np.reshape(net_glob.fc3.weight.data.clone().cpu().numpy(), (1, 50 * 1)))

        val_acc, val_RMSE, vali_loss = test_img(net_glob, dataset_test, args, criterion)
        # acc_test, loss_test = test_img(net_glob, dataset_test, args, criterion)

        if val_RMSE < min_val_RMSE:
            min_val_RMSE = val_RMSE
            min_R_epoch = iter

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Average vali_loss {:.3f}, Accuracy {:.2f}%'
              .format(iter, loss_avg, vali_loss, val_acc))
        loss_train.append(loss_avg)
        valid_loss.append(vali_loss)
        val_accuracy.append(val_acc.item()/100.0)
        # pdb.set_trace()

    print("####################################################################### \n")
    print("Training complete \n")
    print(" min RMSE = {} at {} epoch \n".format(min_val_RMSE, min_R_epoch))
    print("####################################################################### \n")

    # testing the validation set
    test_img(net_glob, dataset_test, args, criterion, eval = True)

    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                      'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '14'}

    # plot weights
    weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
    
    np.savetxt(output_path + '/who.csv', weight_ho, delimiter = ',')
    np.savetxt(output_path + '/val_acc.csv', val_accuracy, delimiter = ',')
    np.savetxt(output_path + '/train_loss.csv', loss_train, delimiter = ',')
    np.savetxt(output_path + '/val_loss.csv', valid_loss, delimiter = ',')

    
    
    # weights_ho_num = int(np.shape(weight_ho)[1])
    # for i in range(0, weights_ho_num):
        # plt.plot(weight_ho[:, i])
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylabel("weights from hidden to output layer", **axis_font)
    # plt.xlabel("communication round", **axis_font)
    # plt.xlim(0, args.epochs)
    # plt.rcParams['agg.path.chunksize'] = 10000
    # plt.savefig(output_path + '/who.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), np.log(loss_train), linewidth = 2, label = 'training loss', c='blue' )
    # plt.plot(range(len(valid_loss)), np.log(valid_loss), linewidth = 2, linestyle = '--',
             # label = 'validation loss' , c='green')
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.ylabel('log(MSE Loss)', **axis_font)
    # plt.xlabel('communication round', ** axis_font)
    # # plt.title ("Training Loss vs training rounds using FedAvg", **title_font)
    # plt.legend()
    # plt.xlim(0, args.epochs)
    # # plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), dpi = 600)
    # plt.savefig(output_path + '/log_loss.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), (loss_train), linewidth=3, label='training loss', c='blue')
    # plt.plot(range(len(valid_loss)), (valid_loss), linewidth=3, label='validation loss', c='green')
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.ylabel('MSE Loss', **axis_font)
    # plt.xlabel('communication round', **axis_font)
    # # plt.title ("Training Loss vs training rounds using FedAvg", **title_font)
    # plt.legend()
    # plt.xlim(0, args.epochs)
    # # plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), dpi = 600)
    # plt.savefig(output_path + '/loss.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(val_accuracy)), (val_accuracy), linewidth=2, label='validation accuracy', c='blue')
    # plt.grid(linestyle='-', linewidth=0.5)
    # plt.ylabel('validation accuracy', **axis_font)
    # plt.xlabel('communication round', **axis_font)
    # # plt.title ("Training Loss vs training rounds using FedAvg", **title_font)
    # plt.legend()
    # plt.xlim(0, args.epochs)
    # # plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid), dpi = 600)
    # plt.savefig(output_path + '/val_acc.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # # uncomment below to plot input to hidden weights
    # weight_ih = np.reshape(weight_ih, (np.shape(weight_ih)[0], np.shape(weight_ih)[2]))
    # weights_ih_num = int(np.shape(weight_ih)[1])
    # for i in range(0, weights_ih_num):
    #     plt.plot(weight_ih[:, i])
    # plt.grid(linestyle='-', linewidth = 0.5)
    # plt.xticks(fontsize = 12)
    # plt.yticks(fontsize = 12)
    # plt.ylabel("weights from input to hidden layer", **axis_font)
    # plt.xlabel("communication round", **axis_font)
    # plt.xlim(0, args.epochs)
    # plt.rcParams['agg.path.chunksize'] = 10000
    # plt.savefig('wih.png', dpi=600, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # # testing
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))

