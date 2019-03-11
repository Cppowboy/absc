import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, query, keys, values, mode='concept'):
        '''
        query: batch * dim_q / batch * len1 * dim_q
        keys: batch * len * dim_k 
        values: batch * len * dim_v
        '''
        if mode == 'target':
            g = torch.relu(self.wx(keys) + self.wh(query)[:, None, :])  # batch, len, middle
            e = torch.softmax(self.w(g), -1)  # batch, len, 1
            output = torch.bmm(values.transpose(1, 2), e)  # batch, dim_value, 1
            output = torch.squeeze(output, -1)  # batch, dim_value
        elif mode == 'concept':
            g = torch.relu(self.wx(keys)[:, None, :, :] + self.wh(query)[None, :, None, :]) # batch, num_concept, len, middle 
            e = torch.softmax(self.w(g), 2) # batch, num_concept, len, 1
            e = torch.squeeze(e, -1) # batch, num_concept, len 
            output = torch.bmm(e, values) # batch, num_concept, dim_value
        return output


class Hair(nn.Module):
    def __init__(self, kernel_size, num_channel, dim_middle, num_concept,
                 dim_concept, num_classification, maxlen, dim_word, dropout_rate, wordemb, targetemb):
        super(Hair, self).__init__()
        self.kernel_size = kernel_size
        self.num_channel = num_channel
        self.dim_middle = dim_middle
        self.num_concept = num_concept
        self.dim_concept = dim_concept
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.dim_word = dim_word
        self.dropout_rate = dropout_rate
        self.emb_matrix = self.init_emb(wordemb)
        self.target_matrix = self.init_emb(targetemb)
        self.concept_matrix = nn.Parameter(torch.tensor(
            np.random.uniform(size=[num_concept, dim_concept]), dtype=torch.float32, requires_grad=True))
        self.kernel_sizes = [3, 4, 5]
        self.conv_list = nn.ModuleList([nn.Conv1d(dim_word, num_channel, k) for k in self.kernel_sizes])
        # self.conv = nn.Conv1d(dim_word, num_channel, kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_channel * len(self.kernel_sizes), num_classification, True)
        self.stock_linear = nn.Linear(dim_word, num_channel, True)

    def forward(self, sent, target, lens):
        '''
        sent: batch * maxlen 
        target: batch 
        lens: batch 
        '''
        # batch = sent.shape[0]
        x = self.emb_matrix(sent).view(
            sent.shape[0], sent.shape[1], -1)  # batch * maxlen * dim_word
        target_x = self.target_matrix(target).view(target.shape[0], -1)  # batch * dim_word
        stock_coefficient = torch.tanh(self.stock_linear(target_x))
        h = [F.relu(conv(x.transpose(1, 2))) for conv in self.conv_list]  # batch, num_channel, len
        r = [F.max_pool1d(a, a.size(2)).squeeze(2) * stock_coefficient for a in h]
        r = torch.cat(r, -1)
        r = self.dropout(r)
        logit = self.linear(r)
        return logit

    def init_emb(self, embedding):
        emb_matrix = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)
        return emb_matrix
