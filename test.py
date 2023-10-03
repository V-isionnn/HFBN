from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import DataLoadAdni
from model_hierar import model_hierar
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import time
def _init_():
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists('log/'+args.exp_name):
        os.makedirs('log/'+args.exp_name)
    if not os.path.exists('log/'+args.exp_name+'/'+'models'):
        os.makedirs('log/'+args.exp_name+'/'+'models')



def test(args, fold):
    test_loader = DataLoader(DataLoadAdni(partition='test', partroi=args.partroi,  fold=fold+1, choose_data=args.data_choose), num_workers=0,
                            batch_size=30, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = model_hierar(args).to(device)
    if not os.path.exists(args.pretrained_path):
        print("file does not exist!")
        return

    para = torch.load(args.pretrained_path)
    print(para["acc"])
    model.load_state_dict(para['model_state_dict'])
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []

    total_time = 0.0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        batch_size = data.size()[0]
        start_time = time.time()
        logits= model(data)
        logits = logits.squeeze(1)
        label = label.to(torch.float32)
        value, preds = torch.max(logits.data, 1)
        end_time = time.time()
        total_time += (end_time - start_time)
        count += batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    print('test total time is', total_time)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)

    score1 = metrics.confusion_matrix(test_true, test_pred)
    TN = score1[0][0]
    FN = score1[1][0]
    FP = score1[0][1]
    TP = score1[1][1]
    SPE = TN / (TN + FP)
    SEN = TP / (TP + FN)
    all = [test_acc, SPE, SEN]
    return test_acc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HFBN')
    parser.add_argument('--exp_name', type=str, default='train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=299, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=int, default=0,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N')
    parser.add_argument('--adni', type=int, default=2, choices=[2,3])
    parser.add_argument('--kernel', type=int, default=9)
    parser.add_argument('--partroi', type=int, default=270)
    parser.add_argument('--log_dir', type=str, default='output', help='experiment root')
    parser.add_argument('--Gbias', type=bool, default=False, help='if bias ')
    parser.add_argument('--num_pooling', type=int, default=1, help=' ')
    parser.add_argument('--embedding_dim', type=int, default=90, help=' ')
    parser.add_argument('--assign_ratio', type=float, default=0.35, help=' ')
    parser.add_argument('--assign_ratio_1', type=float, default=0.35, help=' ')
    parser.add_argument('--mult_num', type=int, default=8, help=' ')
    parser.add_argument('--data_choose', type=str, default='adni3', help='choose model')
    parser.add_argument('--fold_list',  default=[3], help='')
    parser.add_argument('--pretrained_path', type=str, default=
    os.path.join(os.getcwd(),"log/pretrainModel/train/models/model.t7"))



    allaccu = []
    args = parser.parse_args()
    for i in args.fold_list:
        fold = i
        args = parser.parse_args()
        _init_()

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        acc = test(args,  fold)
        print("the model HFBN test result:", acc)
