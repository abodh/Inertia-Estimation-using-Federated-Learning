import matplotlib
matplotlib.use('Agg') # non-interactive backend to save the file
import matplotlib.pyplot as plt
import pdb # debugging

import torch
import numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from utils import args_parser # if run from here default arguments will be passed
from models.network import MLP


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # seeding the generator for reproducibility
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    # load dataset
    if args.dataset == 'mnist':
        #transforming the data to tensors and then normalizing the data
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(), # Convert to pytorch tensors
                       transforms.Normalize((0.1307,), (0.3081,)) # data normalization  --> mean=(0.1307,), std=(0.3081,)
                   ]))
        img_size = dataset_train[0][0].shape # size of image, for eg. the output would be --> torch.Size([1, 28, 28])
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            # pdb.set_trace();
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    # pdb.set_trace()

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum) # standard gradient descent
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)

    # pdb.set_trace()
    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)

            # clears x.grad for every parameter xin the optimizer
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)

            # accumulates the gradient for each parameter --> x.grad += dloss/dx
            loss.backward()

            # performs parameter update on the current gradient --> x += -lr * x.grad
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    ############### Plotting the errors #####################

    # Set the font dictionaries (for plot title and axis titles)
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}  # Bottom vertical alignment for more space
    axis_font = {'fontname': 'Arial', 'size': '14'}

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs', **axis_font)
    plt.ylabel('train loss', **axis_font)
    plt.title("Training Loss vs Epochs for MLP based MNIST", **title_font)
    plt.grid(linestyle='-', linewidth=0.5)
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs), dpi = 600)

    ########### testing  ########
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)