import argparse
import json
import pickle
import torch

from com.utils import shuffle_data, padding, pad_answer, get_model_parameters
from models.bidaf import BiDAF
from models.dcn import DCN
from models.mwan_full import MwAN_full
from models.mwan_ori import MwAN

opts=json.load(open("models/mwan_config.json"))
parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='net/mwan_f.pt',
                    help='path to save the final model')
args = parser.parse_args()


def train(epoch, net,train_dt, opt, best):
    net.train()
    data = shuffle_data(train_dt, 1)
    total_loss = 0.0
    for num, i in enumerate(range(0, len(data), opts["batch"])):
        one = data[i:i + opts["batch"]]
        query, _ = padding([x[0] for x in one], max_len=50)
        passage, _ = padding([x[1] for x in one], max_len=350)
        answer = pad_answer([x[2] for x in one])
        query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
        if args.cuda:
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        opt.zero_grad()
        loss = net([query, passage, answer, True])
        loss.backward()
        total_loss += loss.item()
        opt.step()
        if (num + 1) % opts["log_interval"] == 0:
            print ('|------epoch {:d} train error is {:f}  eclipse {:.2f}% best {}------|'.format(epoch,
                                                                                         total_loss / opts["log_interval"],
                                                                                         i * 100.0 / len(data),best))
            total_loss = 0


def test(net, valid_data):
    net.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(valid_data), opts["batch"]):
            print("{} in {}".format(i, len(valid_data)))
            one = valid_data[i:i + opts["batch"]]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=500)
            answer = pad_answer([x[2] for x in one])
            query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = net([query, passage, answer, False])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(opts["epoch"]):
        train(epoch,model,train_data,optimizer,best)
        acc = test(net=model,valid_data=dev_data)
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        print ('epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))


if __name__ == '__main__':
    # model = MwAN_full(vocab_size=opts["vocab_size"], embedding_size=opts["emb_size"], encoder_size=opts["hidden_size"],
    #                   drop_out=opts["dropout"])  # 16821760 | 5 epoch -> 69.12 on test 69.36 on dev

    # model = MwAN(vocab_size=opts["vocab_size"], embedding_size=opts["emb_size"], encoder_size=opts["hidden_size"],
    #                   drop_out=opts["dropout"]) # 14751104

    model = DCN(vocab_size=opts["vocab_size"], embedding_size=opts["emb_size"], encoder_size=opts["hidden_size"],
                      drop_out=opts["dropout"])  # 13770496

    # model = BiDAF(vocab_size=opts["vocab_size"], embedding_size=opts["emb_size"], encoder_size=opts["hidden_size"],
    #             drop_out=opts["dropout"])  #
    print('Model total parameters:', get_model_parameters(model))
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adamax(model.parameters())

    with open(opts["data"] + 'train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(opts["data"] + 'dev.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    dev_data = sorted(dev_data, key=lambda x: len(x[1]))

    print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))

    main()