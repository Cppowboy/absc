import torch
import torch.nn as nn
import torch.nn.functional as F


class GCM(nn.Module):
    def __init__(self, dim_word, num_channel, kernel_size, aspect_kernel_size, num_layer, num_class, wordmat,
                 dropout_rate, device):
        super().__init__()
        self.dim_word = dim_word
        self.num_channel = num_channel
        self.kernel_size = kernel_size
        self.aspect_kernel_size = aspect_kernel_size
        self.num_layer = num_layer
        self.num_class = num_class
        self.wordemb = nn.Embedding.from_pretrained(torch.from_numpy(wordmat).float(), freeze=True)
        self.device = device
        # asepct conv
        self.conv3 = nn.Conv1d(dim_word, num_channel, aspect_kernel_size, padding=(aspect_kernel_size - 1) // 2)
        self.attention = Attention(dim_word, num_channel)
        # conv
        self.conv1 = nn.Conv1d(dim_word, num_channel, kernel_size, padding=(kernel_size - 1) // 2)
        # gate conv
        self.aspect_linear = nn.Linear(num_channel, num_channel, bias=True)
        self.conv2 = nn.Conv1d(dim_word + num_channel, num_channel, kernel_size, padding=(kernel_size - 1) // 2)
        # highway
        self.highway = HighWay(num_channel)
        # output linear
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_channel, num_class, bias=True)

    def forward(self, context_ids, aspect_ids):
        batch, l1 = context_ids.shape
        batch, l2 = aspect_ids.shape
        context = self.wordemb(context_ids).view(batch, l1, -1)
        aspect = self.wordemb(aspect_ids).view(batch, l2, -1)
        m = context
        # single layer
        a = F.relu(self.conv3(aspect.transpose(1, 2)).transpose(1, 2))  # batch, l2, num_channel
        a = self.attention(m, a)  # batch, l1, num_channel
        s = torch.tanh(self.conv1(m.transpose(1, 2)))  # batch, num_channel, l1
        g = F.relu(self.conv2(torch.cat([m, self.aspect_linear(a)], -1).transpose(1, 2)))
        m = s * g  # batch, l1, num_channel
        # highway
        m = self.highway(m.transpose(1, 2))
        # output
        m = m.transpose(1, 2)
        out = F.max_pool1d(m, m.size(2)).squeeze(2)  # batch, num_channel
        out = self.dropout(out)
        return self.linear(out)


class Attention(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()
        self.W = nn.Linear(d1 + d2, d1 + d2, bias=False)
        self.V = nn.Linear(d1 + d2, 1, bias=False)

    def forward(self, keys, values):
        '''
        :param key: batch, l1, d1
        :param values: batch, l2, d2
        :return:
        '''
        l1 = keys.shape[1]
        l2 = values.shape[1]
        a = keys.unsqueeze(2).expand(-1, -1, l2, -1)  # batch, l1, l2, d1
        b = values.unsqueeze(1).expand(-1, l1, -1, -1)  # batch, l1, l2, d2
        s = F.tanh(self.W(torch.cat([a, b], -1)))  # batch, l1, l2, d1 + d2
        alpha = F.softmax(self.V(s).squeeze(-1), -1)  # batch, l1, l2
        out = torch.bmm(alpha, values)  # batch, l1, d2
        return out


class HighWay(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.gate_linear = nn.Linear(dim, dim)

    def forward(self, x):
        h = F.relu(self.linear(x))
        g = torch.sigmoid(self.gate_linear(x))
        return h * g + x * (1 - g)
