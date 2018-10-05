import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BiDAF(nn.Module):  # param: 16821760
    def __init__(self, vocab_size, embedding_size, encoder_size, drop_out=0.2):
        super(BiDAF, self).__init__()
        self.drop_out = drop_out
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)

        self.a_encoder = nn.LSTM(input_size=embedding_size, hidden_size=int(encoder_size / 2), batch_first=True,
                                 bias=False,bidirectional=True)
        self.q_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                 bias=False,bidirectional=True)
        self.d_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                       bias=False,bidirectional=True)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.W_Q = nn.Linear(encoder_size, encoder_size, bias=True)

        """
       prediction layer
       """
        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)  # embedding初始化为-0.1~0.1之间
        for module in self.modules():
            if isinstance(module, nn.Linear):  # 用0.1来限制，初始化所有nn.Linear的权重
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, is_train] = inputs
        # Embedding
        q_emb = self.embedding(query)
        d_emb = self.embedding(passage)
        a_emb = self.embedding(answer)
        # Layer1: Encoding Layer
        # Encoding a
        a_embedding, _ = self.a_encoder(a_emb.view(-1, a_emb.size(2), a_emb.size(3)))  # （3b,a,2h)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,2h)-> (3b,1,2h)
        a_embedding = a_output.view(a_emb.size(0), 3, -1)  # (b,3,2h)

        #   1 Contextual Embedding Layer
        U, _ = self.q_encoder(q_emb)
        U = F.dropout(U, self.drop_out)  # (b,q,2h)

        H, _ = self.d_encoder(d_emb)
        H = F.dropout(H, self.drop_out)  # (b,p,2h)

        #   2  Attention Flow Layer.
        S=H.bmm(U.transpose(2,1)) # (b,p,q)
        print("S: {}".format(np.shape(S)))
        context2ques_sim=F.softmax(S,dim=2) # on q
        context2ques_att=context2ques_sim.bmm(H) # (b,p,q) (b,q,2h)->(b,p,2h)


        # maxP_si