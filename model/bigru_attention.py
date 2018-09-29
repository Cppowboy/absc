import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_class, wordmat, device):
        super().__init__()
        self.context_enc = nn.GRU(dim_word, dim_hidden, bidirectional=True, batch_first=True)
        # self.aspect_enc = nn.GRU(dim_word, dim_hidden, bidirectional=True, batch_first=True)
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(wordmat))
        self.att = Attention(dim_word)
        self.W = nn.Linear(dim_word, dim_word)
        self.V = nn.Linear(dim_word, 1)
        self.output = nn.Linear(dim_hidden * 2, num_class)

    def forward(self, context_ids, context_masks, aspect_ids, aspect_masks):
        context = self.wordemb(context_ids)
        aspect = self.wordemb(aspect_ids)
        context_ = self.att(context, aspect)
        aaaa = torch.tanh(self.W(context) + context_)
        alpha = F.softmax(self.V(aaaa), 1)  # batch, l, 1
        senti, _ = self.context_enc(context)
        senti = torch.bmm(alpha.transpose(1, 2), senti).squeeze(1)  # batch, 1, dim_hidden * 2
        logit = self.output(senti)
        return logit


class Attention(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.Wc = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.Wa = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.V = nn.Linear(dim_hidden, 1, bias=False)

    def forward(self, context, aspect):
        '''
        :param context: batch, l1, d
        :param aspect: batch, l2, d
        :return: batch, l1, d
        '''
        s = self.Wc(context).unsqueeze(2) + self.Wa(aspect).unsqueeze(1)  # batch, l1, l2, d
        alpha = self.V(torch.tanh(s))  # batch, l1, l2, 1
        alpha = F.softmax(alpha, 2).squeeze(-1)  # batch, l1, l2
        out = torch.bmm(alpha, aspect)  # batch, l1, d
        return out
