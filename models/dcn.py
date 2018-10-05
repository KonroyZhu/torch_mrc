import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DCN(nn.Module):  # param: 16821760
    def __init__(self, vocab_size, embedding_size, encoder_size, drop_out=0.2):
        super(DCN, self).__init__()
        self.drop_out = drop_out
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)

        self.a_encoder = nn.LSTM(input_size=embedding_size, hidden_size=int(encoder_size / 2), batch_first=True,
                                 bias=False,bidirectional=True)
        self.q_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                 bias=False)
        self.d_encoder = nn.LSTM(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                       bias=False)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)

        self.W_Q = nn.Linear(encoder_size, encoder_size, bias=True)

        self.U_lstm = nn.LSTM(input_size=4*encoder_size,hidden_size=encoder_size,batch_first=True,bidirectional=True,bias=False)

        """
        prediction layer
        """
        self.Wq = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear( encoder_size, encoder_size, bias=False)
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
        a_embedding, _ = self.a_encoder(a_emb.view(-1, a_emb.size(2), a_emb.size(3)))  # （3b,a,h)
        a_score = F.softmax(self.a_attention(a_embedding), 1)  # (3b,a,1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()  # (3b,1,a) bmm (3b,a,h)-> (3b,1,h)
        a_embedding = a_output.view(a_emb.size(0), 3, -1)  # (b,3,h)

        #   DYNAMIC COATTENTION NETWORKS
        #   1 DOCUMENT AND QUESTION ENCODER
        Q_, _ = self.q_encoder(q_emb)
        Q_ = F.dropout(Q_, self.drop_out)  # (b,q,h)
        Q = F.tanh(self.W_Q(Q_))

        D, _ = self.d_encoder(d_emb)
        D = F.dropout(D, self.drop_out)  # (b,d,h)

        #   2 COATTENTION ENCODER
        L = D.bmm(Q.transpose(2, 1))  # (b,d,h) bmm (b,h,q)
        # print("L: {}".format(np.shape(L)))  # (b,d,q)
        AQ=F.softmax(L,dim=2) # (b,d,q)
        AD=F.softmax(L.transpose(2,1),dim=2) # (b,q,d)
        # print("AQ: {}".format(np.shape(AQ)))
        # print("AD: {}".format(np.shape(AD)))

        CQ=D.transpose(2,1).bmm(AQ) # (b,d,h) (b,d,q) -> (b,h,q)
        # print("CQ: {}".format(np.shape(CQ)))

        Q_CQ=torch.cat([Q,CQ.transpose(2,1)],2) # (b.q.4h)
        # print("Q_CQ: {}".format(np.shape(Q_CQ)))

        CD=AD.transpose(2,1).bmm(Q_CQ)  # (b,d,q) (b,q,4h) -> (b.d.4h)
        # print("CD: {}".format(np.shape(CD)))

        D_CD=torch.cat([torch.cat([D,D],2),CD],2) # (b,d,4h)
        # print("D_CD: {}".format(np.shape(D_CD)))

        U,_ = self.U_lstm(D_CD)
        # print("U: {}".format(np.shape(U)))  # (b,d,2h)

        # 3: Prediction Layer
        # Layer4: Prediction Layer
        sj = self.vq(torch.tanh(self.Wq(Q))).transpose(2, 1)  # (b,q,h) (h,h) (h,1) -> (b,q,1) -> (b,1,q)
        rq = F.softmax(sj, 2).bmm(Q)  # (b,1,q) (b,q,h) -> (b,1,h)
        sj = F.softmax(self.vp(self.Wp1(U) + self.Wp2(rq)).transpose(2, 1), 2)
        rp = sj.bmm(U)
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)), self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
            return score.argmax(1)
        loss = -torch.log(score[:, 0]).mean()
        return loss
