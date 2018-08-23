import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
from sklearn.metrics.classification import accuracy_score, precision_score, recall_score, f1_score


class FullConnect(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullConnect, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size))
        u = 1 / sqrt(input_size)
        torch.nn.init.uniform_(self.weight, -u, u)
        self.bias.data.zero_()

    def forward(self, x):
        logit = torch.matmul(x, self.weight) + self.bias
        return logit


class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_param()

    def forward(self, x, h_fore, c_fore):
        input = torch.cat([h_fore, x], dim=-1)
        i, f, o, c = torch.matmul(input, self.Wi) + self.Bi, torch.matmul(input, self.Wf) + self.Bf, \
                     torch.matmul(input, self.Wo) + self.Bo, torch.matmul(input, self.Wc) + self.Bc
        i, f, o, c = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(c)
        o = F.threshold(o, threshold=0.4, value=0)
        c = f * c_fore + i * c
        h = o * torch.tanh(c)
        return h, c

    def init_param(self):
        u = 1 / sqrt(self.hidden_size)
        dim_param = self.input_size + self.hidden_size
        self.Wi = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wf = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wo = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Wc = nn.Parameter(torch.Tensor(dim_param, self.hidden_size))
        self.Bi = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bf = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bo = nn.Parameter(torch.Tensor(self.hidden_size))
        self.Bc = nn.Parameter(torch.Tensor(self.hidden_size))
        torch.nn.init.uniform_(self.Wi, -u, u)
        torch.nn.init.uniform_(self.Wf, -u, u)
        torch.nn.init.uniform_(self.Wo, -u, u)
        torch.nn.init.uniform_(self.Wc, -u, u)
        self.Bi.data.zero_()
        self.Bf.data.zero_()
        self.Bo.data.zero_()
        self.Bc.data.zero_()


def get_mask(maxlen, lens):
    device = lens.device
    batch = lens.shape[0]
    idx = torch.range(0, maxlen - 1, 1).to(device)
    idx = torch.stack([idx] * batch)
    mask = idx < lens[:, None].float()
    mask = mask.float()
    return mask


def get_acc(logit, labels):
    correct = torch.sum(torch.argmax(logit, dim=-1) == labels)
    acc = correct.float() / len(labels)
    return acc


def masked_softmax(A, mask):
    # matrix A is the one you want to do mask softmax at dim=1
    A_max = torch.max(A, dim=1, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=1, keepdim=True) + 1e-10)
    return A_softmax


def get_score(a, b_max):
    a_max = np.argmax(a, axis=-1)
    acc = accuracy_score(a_max, b_max)
    p = precision_score(a_max, b_max, average='macro')
    r = recall_score(a_max, b_max, average='macro')
    f1 = f1_score(a_max, b_max, average='macro')
    return acc, p, r, f1


if __name__ == '__main__':
    lens = torch.from_numpy(np.array([3, 4, 5, 6, 3])).long().cuda()
    mask = get_mask(10, lens)
    print(mask)
