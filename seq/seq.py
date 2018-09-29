import torch
import torch.nn as nn
import torch.nn.functional as F


class Sequence(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_class, wordmat, device):
        super().__init__()
        self.context_rnn1 = nn.GRU(dim_word, dim_hidden, bidirectional=True, batch_first=True, bias=True)
        # self.context_rnn2 = nn.GRU(dim_hidden * 2, dim_hidden, bidirectional=True, batch_first=True, bias=True)
        self.linear = nn.Linear(dim_hidden * 2, num_class)
        self.num_class = num_class
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(wordmat))
        self.device = device

    def forward(self, context_ids, context_labels, aspect_mask, aspect_pos):
        context = self.wordemb(context_ids)
        h, _ = self.context_rnn1(context)
        # h, _ = self.context_rnn2(h)
        logit = self.linear(h)
        y = logit * aspect_pos.unsqueeze(2)
        y = torch.sum(y, 1) / (torch.sum(aspect_pos.unsqueeze(2), 1) + 0.0001)
        loss = F.cross_entropy(logit.view(-1, self.num_class), context_labels.view(-1), reduction='none')
        loss = torch.sum(loss * aspect_mask.unsqueeze(2)) / (torch.sum(aspect_mask) + 0.0001)
        return loss, y
