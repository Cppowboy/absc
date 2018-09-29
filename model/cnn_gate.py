import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim_word, num_channel, ks, num_class, wordmat, device):
        super().__init__()
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(wordmat))
        self.context_conv = nn.Conv1d(dim_word, num_channel, ks, padding=(ks - 1) // 2)
        self.gate_conv = nn.Conv1d(dim_word * 2, num_channel, ks, padding=(ks - 1) // 2)
        self.att = FullyAwareAttention(dim_word)
        self.linear = nn.Linear(num_channel, num_class)
        self.device = device

    def forward(self, context_ids, context_masks, aspect_ids, aspect_masks):
        context = self.wordemb(context_ids)
        aspect = self.wordemb(aspect_ids)
        gate_input = torch.cat([context, self.att(context, aspect, aspect, aspect_masks)], -1)
        sentiment = torch.tanh(self.context_conv(context.transpose(1, 2)))
        gate = F.relu(self.gate_conv(gate_input.transpose(1, 2)))
        out = sentiment * gate
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.linear(out)
        return out


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
