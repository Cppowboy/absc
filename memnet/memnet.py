import torch
import torch.nn as nn
import torch.nn.functional as F
from memnet.basic import get_mask, masked_softmax


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

    def forward(self, sent_ids, aspect_ids, sent_lens, aspect_lens, position_weights):
        '''
        :param sent_ids: batch, l1
        :param aspect_ids: batch, l2
        :param sent_lens: batch
        :param aspect_lens: batch
        :param position_weights: batch, l1
        :return:
        '''
        l1 = sent_ids.shape[1]
        sent_x = self.wordemb(sent_ids)  # batch, l1, dim_word
        aspect_x = self.wordemb(aspect_ids)  # batch, l2, dim_word
        mask = get_mask(l1, sent_lens)
        vec = torch.sum(aspect_x, 1) / aspect_lens.unsqueeze(-1).float()  # batch, dim_word
        # m = sent_x * position_weights  # batch, l1, dim_word
        # m = sent_x + position_weights  # batch, l1, dim_word
        m = sent_x  # the performance will get worse when add position weights
        for _ in range(self.num_hop):
            alpha = F.tanh(
                self.Watt(torch.cat([m, vec.unsqueeze(1).expand(-1, l1, -1)], -1)))  # batch, l1, 1
            # alpha = F.softmax(alpha, 1)
            # alpha = masked_softmax(alpha, mask.unsqueeze(-1), 1)
            alpha = masked_softmax(alpha * position_weights.unsqueeze(-1), mask.unsqueeze(-1), 1)
            out = torch.bmm(alpha.transpose(1, 2), m)  # batch, 1, dim_word
            out = out.squeeze(1)  # batch, dim_word
            vec = self.W(vec) + out  # batch, dim_word
        logit = self.output(vec)  # batch, num_class
        return logit

    def init_emb(self, mat):
        emb = nn.Embedding.from_pretrained(torch.from_numpy(mat).float(), freeze=True)  # no updating
        return emb
