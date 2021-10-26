import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, criterion, eval = False):
    with torch.no_grad():
        val_correct = []
        val_loss_func = []
        n_items = 0
        net_g.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=args.bs)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            n_items += len(target)
            data, target = data.to(args.device), target.to(args.device)
            # if args.gpu != -1:
            #     data, target = data.cuda(), target.cuda()

            # data, target = data.to(args.device), target.to(args.device)
            target = target.float().view(-1, 1)

            oupt = net_g(data.float())
            # log_probs = net_g(data)

            # sum up batch loss
            # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            loss = criterion(oupt,target)
            n_correct = torch.sum((torch.abs(oupt - target) < torch.abs(0.1 * target)))
            val_correct.append(n_correct)
            val_loss_func.append(loss)

        loss_func = sum(val_loss_func) / len(val_loss_func)
        result = (sum(val_correct) * 100.0 / n_items)
        RMSE_loss = torch.sqrt(loss_func)

        # observing the result when set to eval mode
        if (eval):
            print('Predicted test output for random batch = {}, actual output = {} with accuracy of {:.2f}% '
                  'and RMSE = {:.6f}'.format(oupt, target, result, RMSE_loss))
        return result, RMSE_loss, loss_func

        #     # get the index of the max log-probability
        #     y_pred = log_probs.data.max(1, keepdim=True)[1]
        #     correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        #
        # test_loss /= len(data_loader.dataset)
        # accuracy = 100.00 * correct / len(data_loader.dataset)
        # if args.verbose:
        #     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        #         test_loss, correct, len(data_loader.dataset), accuracy))
        # return accuracy, test_loss

