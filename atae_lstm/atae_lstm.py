import torch
import torch.nn as nn
import torch.nn.functional as F
from atae_lstm.basic import LstmCell, FullConnect, get_mask, masked_softmax
from math import sqrt


class ABSA_Atae_Lstm(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_classification, maxlen, wordemb, targetemb, device):
        super(ABSA_Atae_Lstm, self).__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.init_param()
        self.emb_matrix = self.init_emb(wordemb)
        self.target_matrix = self.init_emb(targetemb)
        self.device = device

    def forward(self, sent, target, lens):
        batch = sent.shape[0]
        x = self.emb_matrix(sent).view(sent.shape[0], sent.shape[1], -1)
        target_x = self.target_matrix(target).view(target.shape[0], -1)
        h, c = torch.zeros(batch, self.dim_hidden).to(self.device), torch.zeros(batch, self.dim_hidden).to(self.device)
        mask = get_mask(self.maxlen, lens)
        h_list = []
        for t in range(self.maxlen):
            input = torch.cat([x[:, t, :], target_x], dim=-1)
            _h, _c = self.lstmcell(input, h, c)
            m = mask[:, t]
            h = m[:, None] * _h + (1 - m)[:, None] * h
            c = m[:, None] * _c + (1 - m)[:, None] * c
            h_list.append(h)
        H = torch.stack(h_list, dim=1)

        # attention
        matrix_aspect = torch.zeros_like(H).float() + target_x[:, None, :]
        hhhh = torch.cat([torch.matmul(H, self.Wh), torch.matmul(matrix_aspect, self.Wv)], dim=-1)
        M_tmp = torch.tanh(hhhh)
        alpha_tmp = masked_softmax(torch.matmul(M_tmp, self.w), mask)
        r = torch.bmm(alpha_tmp[:, None, :], H).squeeze()
        h = torch.tanh(torch.matmul(r, self.Wp) + torch.matmul(H[:, -1, :], self.Wx))
        h = self.drop(h)
        logit = self.linear(h)
        return logit

    def init_param(self):
        self.lstmcell = LstmCell(input_size=self.dim_word * 2, hidden_size=self.dim_hidden)
        self.linear = FullConnect(self.dim_hidden, self.num_classification)
        self.Wh = nn.Parameter(torch.Tensor(self.dim_hidden, self.dim_hidden))
        self.Wv = nn.Parameter(torch.Tensor(self.dim_word, self.dim_word))
        self.w = nn.Parameter(torch.Tensor(self.dim_hidden + self.dim_word, ))
        self.Wp = nn.Parameter(torch.Tensor(self.dim_hidden, self.dim_hidden))
        self.Wx = nn.Parameter(torch.Tensor(self.dim_hidden, self.dim_hidden))
        u = 1 / sqrt(self.dim_hidden)
        nn.init.uniform_(self.Wh, -u, u)
        nn.init.uniform_(self.Wv, -u, u)
        nn.init.uniform_(self.w, -u, u)
        nn.init.uniform_(self.Wp, -u, u)
        nn.init.uniform_(self.Wx, -u, u)
        self.drop = nn.Dropout(0.5)

    def init_emb(self, embedding):
        num_word, dim_word = embedding.shape
        emb_matrix = nn.Embedding(num_word, dim_word)
        emb_matrix.weight = nn.Parameter(torch.from_numpy(embedding).float())
        return emb_matrix
