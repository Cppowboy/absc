import torch
import torch.nn as nn
import torch.nn.functional as F
from tnet.basic import FullConnect


class TNet(nn.Module):
    def __init__(self, dim_word, dim_hidden, kernel_size, num_channel, num_class, cpt_num, dropout_rate,
                 word_mat, device):
        super().__init__()
        self.emb = Embedding(dim_word, word_mat)
        self.sent_encoder = Encoder(dim_word, dim_hidden, device)
        self.target_encoder = Encoder(dim_word, dim_hidden, device)
        self.cpt = CPT_AS(2 * dim_hidden, 2 * dim_hidden)
        self.cpt_num = cpt_num
        self.convfeat = ConvFeatExtractor(2 * dim_hidden, kernel_size=kernel_size, channel=num_channel)
        self.output = Output(num_channel, num_class)
        self.sent_dropout = nn.Dropout(dropout_rate)
        self.aspect_dropout = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sent_wids, target_wids, position_weights):
        sent = self.emb(sent_wids)
        sent = self.sent_dropout(sent)
        aspect = self.emb(target_wids)
        aspect = self.aspect_dropout(aspect)
        h = self.sent_encoder(sent)
        ht = self.target_encoder(aspect)
        for _ in range(self.cpt_num):
            h = self.cpt(h, ht, position_weights)
        feat = self.convfeat(h).squeeze()
        feat = self.dropout(feat)
        logit = self.output(feat)
        return logit


class Embedding(nn.Module):
    def __init__(self, dim_word, wordmat):
        super().__init__()
        self.wordemb = self.init_emb(wordmat)

    def forward(self, wids):
        batch, length = wids.shape
        w = self.wordemb(wids).view(batch, length, -1)  # batch, length, dim_word
        return w

    def init_emb(self, mat):
        num, dim = mat.shape
        emb = nn.Embedding(num, dim)
        emb.weight = nn.Parameter(torch.from_numpy(mat).float())
        return emb


class Encoder(nn.Module):
    def __init__(self, dim_word, dim_hidden, device):
        super().__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.device = device
        self.lstm = nn.LSTM(dim_word, dim_hidden, bidirectional=True, batch_first=True)

    def forward(self, x):
        '''
        :param x: batch, length, dim_word
        :return:
        '''
        batch = x.shape[0]
        h, c = torch.zeros([2, batch, self.dim_hidden]).to(self.device), \
               torch.zeros([2, batch, self.dim_hidden]).to(self.device)
        output, _ = self.lstm(x, (h, c))
        return output


class CPT_AS(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = FullConnect(2 * dim_in, dim_out)
        self.as_gate = FullConnect(dim_in, dim_out)
        # self.att_linear = FullConnect(dim_in * 3, 1)

    def forward(self, h, ht, position_weights):
        '''
        :param h: batch, l1, dim
        :param ht: batch, l2, dim
        :param position_weights: batch, l1
        :return:
        '''
        # gate
        t = F.sigmoid(self.as_gate(h))  # batch, l1, dim_hidden
        # target-specific transformation
        # left = torch.zeros_like(ht[:, None, :, :]).to(h.device) + h[:, :, None, :]
        # right = torch.zeros_like(h[:, :, None, :]).to(ht.device) + ht[:, None, :, :]
        # alpha = F.softmax(self.att_linear(torch.cat([left, right, left * right], dim=-1)),
        #                   dim=-1).squeeze()  # (batch, l1, l2)
        alpha = F.softmax(torch.bmm(h, ht.transpose(1, 2)), dim=-1)  # (batch, l1, l2)
        r = torch.bmm(alpha, ht)  # (batch, l1, dim)
        h_new = F.tanh(self.linear(torch.cat([r, h], dim=-1)))  # (batch, l2, dim)
        h_new = h_new * position_weights[:, :, None]
        return t * h_new + (1 - t) * h  # batch, l1, dim_hidden


class ConvFeatExtractor(nn.Module):
    def __init__(self, dim_hidden, kernel_size, channel):
        super().__init__()
        self.conv = nn.Conv1d(dim_hidden, channel, kernel_size)
        nn.init.uniform_(self.conv.weight, -0.01, 0.01)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, h):
        '''
        :param h: batch, len, dim_hidden
        :param mask: batch, len, mask of h
        :return:
        '''
        logit = self.conv(h.transpose(1, 2))  # batch, channel, len
        logit = F.relu(logit)
        logit = F.max_pool1d(logit, logit.size(2))  # batch, channel
        return logit


class Output(nn.Module):
    def __init__(self, in_size, num_class):
        super().__init__()
        self.linear = FullConnect(in_size, num_class)

    def forward(self, input):
        logit = self.linear(input)
        logit = F.softmax(logit, dim=-1)
        return logit
