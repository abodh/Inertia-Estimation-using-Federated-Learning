import argparse
import numpy as np
from torchvision import datasets, transforms
import pdb

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="total rounds of training (communication rounds)")
    parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    # parser.add_argument('--iid', default='False', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--iid', default='False', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=1, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()
    return args

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)     # num of training items per user
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    # dictionary of image indexes
    return dict_users

def mnist_non_iid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    yo = []
    num_shards, num_imgs = 6, 120
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    for idx, (x,y) in enumerate(dataset):
        yo.append(y)

    # labels = dataset.y.numpy()
    labels = np.array(yo)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:].astype(int)

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    # test
    num = 100 # number of users
    d = mnist_non_iid(dataset_train, num)
    print (type(d))