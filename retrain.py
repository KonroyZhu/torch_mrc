import argparse
import json
import pickle

import torch

from models.mwan_full import MwAN_full
from train import test, train


opts=json.load(open("models/mwan_config.json"))
parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--model', type=str, default='net/mwan_f1.pt',  # 上一次训练出来的model
                    help='model path')
parser.add_argument('--save', type=str, default='net/mwan_f2.pt', # 新model的存储位置
                    help='path to save the final model')
args = parser.parse_args()


if __name__ == '__main__':

    with open(args.model, 'rb') as f:
        model = torch.load(f)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters(),lr=2e-4)
    with open(opts["data"] + 'train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(opts["data"] + 'dev.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    dev_data = sorted(dev_data, key=lambda x: len(x[1]))

    print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))

    print("testing...")
    best = test(net=model,valid_data=dev_data)
    print("best: {}".format(best))
    print("training...")
    for epoch in range(opts["epoch"]):
        train(epoch,model,train_data,optimizer,best)
        acc = test(net=model,valid_data=dev_data)
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        print ('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))