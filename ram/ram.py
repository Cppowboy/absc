import torch
import torch.nn as nn
import torch.nn.functional as F


class RAM(nn.Module):
    def __init__(self, dim_word, dim_hidden, dim_episode, num_layer, num_class, wordmat, dropout_rate, device):
        super().__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.dim_episode = dim_episode
        self.num_layer = num_layer
        self.num_class = num_class
        self.device = device
        self.wordemb = self.init_emb(wordmat)
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(dim_word, dim_hidden, bidirectional=True, bias=True, batch_first=True)
        self.att_linear = nn.Linear(dim_hidden * 2 + 1 + dim_episode + dim_word, 1)
        self.grucell = nn.GRUCell(dim_hidden * 2, dim_episode)
        self.output_linear = nn.Linear(dim_episode, num_class)

    def forward(self, sent_ids, aspect_ids, position_weights):
        '''
        :param sent_ids: batch, length
        :param aspect_ids: batch, l2
        :param position_weights: batch, length
        :return:
        '''
        batch, length = sent_ids.shape
        # embedding
        x = self.wordemb(sent_ids)  # batch, length, dim_word
        x = self.dropout(x)
        aspect = self.wordemb(aspect_ids)  # batch, l2, dim_word
        aspect = torch.mean(aspect, 1)
        # bilstm encode
        m, _ = self.encoder(x)  # batch, length, 2 * dim_hidden
        # m = h * position_weights.unsqueeze(2)  # batch, length, 2 * dim_hidden
        # hops
        e = torch.zeros(batch, self.dim_episode).to(self.device)  # batch, dim_episode
        for _ in range(self.num_layer):
            g = self.att_linear(
                torch.cat([m, torch.zeros(batch, length, self.dim_episode).to(self.device) + e.unsqueeze(1),
                           torch.zeros(batch, length, self.dim_word).to(self.device) + aspect.unsqueeze(1),
                           position_weights.unsqueeze(2)],
                          dim=-1))  # batch, length, 1
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), m).squeeze(1)  # batch, 2 * dim_hidden
            e = self.grucell(i, e)
        logit = self.output_linear(e)
        return logit

    def init_emb(self, mat):
        emb = nn.Embedding.from_pretrained(torch.from_numpy(mat).float(), freeze=True)
        return emb
