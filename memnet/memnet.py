import torch
import torch.nn as nn
import torch.nn.functional as F


class MemNet(nn.Module):
    def __init__(self, dim_word, num_class, num_hop, wordmat, device):
        super().__init__()
        self.dim_word = dim_word
        self.wordemb = self.init_emb(wordmat)
        self.device = device
        self.num_hop = num_hop
        self.Watt = nn.Linear(dim_word * 2, 1)
        self.W = nn.Linear(dim_word, dim_word)
        self.output = nn.Linear(dim_word, num_class)

    def forward(self, sent_ids, aspect_ids, position_weights):
        '''
        :param sent_ids: batch, l1
        :param aspect_ids: batch, l2
        :param position_weights: batch, l1
        :return:
        '''
        sent_x = self.wordemb(sent_ids)  # batch, l1, dim_word
        aspect_x = self.wordemb(aspect_ids)  # batch, l2, dim_word
        vec = torch.mean(aspect_x, 1)  # batch, dim_word
        m = sent_x * position_weights.unsqueeze(-1)  # batch, l1, dim_word
        for _ in range(self.num_hop):
            alpha = F.tanh(
                self.Watt(torch.cat([m, vec.unsqueeze(1) + torch.zeros_like(m).to(m.device)], -1)))  # batch, l1, 1
            alpha = F.softmax(alpha, 1)
            vec = torch.bmm(alpha.transpose(1, 2), m)  # batch, 1, dim_word
            vec = self.W(vec.squeeze(1))  # batch, dim_word
        logit = self.output(vec)  # batch, num_class
        return logit

    def init_emb(self, mat):
        emb = nn.Embedding.from_pretrained(torch.from_numpy(mat).float(), freeze=True)  # no updating
        return emb
