import argparse
import pickle
import codecs

import numpy as np
import torch

from com.preprocess import transform_data_to_id
from com.utils import pad_answer, padding

parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='data/4th-submita.txt',
                    help='prediction path')
parser.add_argument('--model', type=str, default='net/mwan_f0.pt',
                    help='model path')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default=False, # FIXME: 服务器上换成TRUE
                    help='use CUDA')

args = parser.parse_args()

def pad_wrong_answer(answer_list):
    # 3680
    # 7136
    # 这两批数据中有alternative answer长度小于3的数据，需要补齐否则无法处理
    # 该方法通过复制ans[0]补齐数据
    padded_list=[]
    for ans in answer_list:
        ll=len(ans)
        if not ll == 3:
            for _ in range(3-ll):
                ans+=[ans[0]]
        padded_list.append(ans)
    padded_list=pad_answer(padded_list)
    return padded_list

with open(args.model, 'rb') as f:
    model = torch.load(f)
if args.cuda:
    model.cuda()

with open(args.word_path, 'rb') as f:
    word2id = pickle.load(f)

# raw_data = seg_data(args.data)
raw_data=pickle.load(open("data/testa_seg.pkl","rb"))
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
data = sorted(data, key=lambda x: len(x[1]))
print ('test data size {:d}'.format(len(data)))


def inference():
    model.eval()
    predictions = []
    id_prediction={}
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            print("{} in {}".format(i,len(data)))
            one = data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=300)
            answer = pad_answer([x[2] for x in one])
            str_words = [x[-1] for x in one]
            ids = [x[3] for x in one]
            answer=pad_wrong_answer(answer)
            query=torch.LongTensor(query)
            passage = torch.LongTensor(passage)
            #print(np.shape(answer))
            answer=torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            for q_id, prediction, candidates in zip(ids, output, str_words):
                id_prediction[q_id]=int(prediction)
                prediction_answer = u''.join(candidates[prediction])
                predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    with open("pkl_records/dev11.pkl","wb") as f:
        pickle.dump(id_prediction,f)
    print('done!')


if __name__ == '__main__':
    inference()