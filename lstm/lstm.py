import torch
import torch.nn as nn
from lstm.basic import LstmCell, FullConnect, get_mask


class ABSA_Lstm(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_classification, maxlen, wordemb, device):
        super(ABSA_Lstm, self).__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.init_param()
        self.emb_matrix = self.init_emb(wordemb)
        self.device = device

    def forward(self, sent, target, lens):
        batch = sent.shape[0]
        x = self.emb_matrix(sent).view(sent.shape[0], sent.shape[1], -1)
        h, c = torch.zeros([batch, self.dim_hidden]).to(self.device), \
               torch.zeros([batch, self.dim_hidden]).to(self.device)
        mask = get_mask(self.maxlen, lens.cpu())
        mask = mask.to(self.device)
        for t in range(self.maxlen):
            _h, _c = self.lstmcell(x[:, t, :], h, c)
            m = mask[:, t]
            h = m[:, None] * _h + (1 - m)[:, None] * h
            c = m[:, None] * _c + (1 - m)[:, None] * c
        logit = self.linear(h)
        return logit

    def init_param(self):
        self.lstmcell = LstmCell(input_size=self.dim_word, hidden_size=self.dim_hidden)
        self.linear = FullConnect(self.dim_hidden, self.num_classification)

    def init_emb(self, embedding):
        num_word, dim_word = embedding.shape
        emb_matrix = nn.Embedding(num_word, dim_word)
        emb_matrix.weight = nn.Parameter(torch.from_numpy(embedding).float())
        return emb_matrix
