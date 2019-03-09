import torch
import torch.nn as nn
import torch.nn.functional as F
from acsa_gcae.basic import LstmCell, FullConnect, get_mask, masked_softmax
from math import sqrt
import numpy as np


class Attention(nn.Module):
    def __init__(self, dim_in, dim_h, dim_middle):
        super(Attention, self).__init__()
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_middle = dim_middle
        self.wx = nn.Linear(dim_in, dim_middle, False)
        self.wh = nn.Linear(dim_h, dim_middle, True)
        self.w = nn.Linear(dim_middle, 1, False)

    def forward(self, query, keys, values):
        '''
        query: batch * dim_q / batch * len1 * dim_q
        keys: batch * len * dim_k 
        values: batch * len * dim_v
        '''
        if query.dim() == 2:
            g = torch.tanh(self.wx(keys) + self.wh(query)
                           [:, None, :])  # batch, len, middle
            e = torch.softmax(self.w(g), -1)  # batch, len, 1
            output = torch.bmm(values.transpose(
                1, 2), e)  # batch, dim_value, 1
            output = torch.squeeze(output, -1)  # batch, dim_value
        elif query.dim() == 3:
            g = torch.tanh(
                self.wx(keys)[:, :, None, :] + self.wh(query[:, None, :, :]))  # batch, len, len1, middle
            e = torch.softmax(self.w(g), 1)  # batch, len, len1, 1
            e = torch.squeeze(e, -1)  # batch, len, len1
            # batch, len1, dim_value
            output = torch.bmm(e.transpose(1, 2), values)
        return output


class Hair(nn.Module):
    def __init__(self, kernel_size, num_channel, dim_middle, num_concept,
                 dim_concept, num_classification, maxlen, dim_word, wordemb, targetemb):
        super(Hair, self).__init__()
        self.kernel_size = kernel_size
        self.num_channel = num_channel
        self.dim_middle = dim_middle
        self.num_concept = num_concept
        self.dim_concept = dim_concept
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.dim_word = dim_word
        self.emb_matrix = self.init_emb(wordemb)
        self.target_matrix = self.init_emb(targetemb)
        self.concept_matrix = nn.Parameter(torch.tensor(
            np.random.uniform(size=[num_concept, dim_concept]), dtype=torch.float32))
        self.conv = nn.Conv1d(dim_word, num_channel, kernel_size)
        self.concept_att = Attention(num_channel, dim_concept, dim_middle)
        self.target_att = Attention(num_channel, dim_word, dim_middle)
        self.linear = nn.Linear(num_channel, num_classification, True)

    def forword(self, sent, target, lens):
        '''
        sent: batch * maxlen 
        target: batch 
        lens: batch 
        '''
        # batch = sent.shape[0]
        x = self.emb_matrix(sent).view(
            sent.shape[0], sent.shape[1], -1)  # batch * maxlen * dim_word
        target_x = self.target_matrix(target).view(
            target.shape[0], -1)  # batch * dim_word
        h = F.tanh(self.conv(x.transpose(1, 2))).transpose(
            1, 2)  # batch, len, num_channel
        # batch, len1, num_channel
        sc = self.concept_att(self.concept_matrix, x, x)
        r = self.target_att(target_x, sc, slice)
        logit = self.linear(r)
        return logit

    def init_emb(self, embedding):
        num_word, dim_word = embedding.shape
        emb_matrix = nn.Embedding(num_word, dim_word)
        emb_matrix.weight = nn.Parameter(torch.from_numpy(embedding).float())
        return emb_matrix
