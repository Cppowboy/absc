import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyAwareAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(dim, dim))
        nn.init.kaiming_normal_(self.W, 0.01)

    def forward(self, x, y, values, mask):
        '''
        :param x: batch, l1, dim
        :param y: batch, l2, dim
        :param values: values
        :return:
        '''
        sx = F.relu(torch.matmul(x, self.W))
        sy = F.relu(torch.matmul(y, self.W))
        s = torch.bmm(sx, sy.transpose(1, 2))
        alpha = masked_softmax(s, mask)
        out = torch.bmm(alpha, values)
        return out


def masked_softmax(A, mask, dim=1):
    '''
    :param A: batch, l1, l2
    :param mask: batch, l2
    :param dim:
    :return:
    '''
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask.unsqueeze(1)  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + 1e-10)
    return A_softmax


class Model(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_class, wordmat, device):
        super().__init__()
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(wordmat))
        self.context_gru = nn.GRU(dim_word, dim_hidden, bidirectional=True, batch_first=True)
        self.gate_gru = nn.GRU(dim_word * 2, dim_hidden, bidirectional=True, batch_first=True)
        self.att = FullyAwareAttention(dim_word)
        self.linear = nn.Linear(dim_hidden * 2, num_class)
        self.device = device

    def forward(self, context_ids, context_masks, aspect_ids, aspect_masks):
        context = self.wordemb(context_ids)

        aspect = self.wordemb(aspect_ids)

        gate_input = torch.cat([context, self.att(context, aspect, aspect, aspect_masks)], -1)
        sentiment, _ = self.context_gru(context)
        gate, _ = self.gate_gru(gate_input)
        out = sentiment * gate
        out = out.transpose(1, 2)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.linear(out)
        return out


class ParamGenerator(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear1 = nn.Linear(dim_in, dim_out * dim_out)
        self.linear2 = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        '''
        :param x: batch, dim_in
        :return:    W batch, dim_out, dim_out
                    B batch, dim_out
        '''
        W = F.relu(self.linear1(x)).view(-1, self.dim_out, self.dim_out)
        B = F.relu(self.linear2(x))
        return W, B
