import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim_word, dim_hidden, num_class, wordmat, device):
        super().__init__()
        self.context_enc = BiRNNVP(dim_word, dim_hidden)
        # self.context_enc = nn.GRU(dim_word, dim_hidden, bidirectional=True, batch_first=True)
        self.wordemb = nn.Embedding.from_pretrained(torch.FloatTensor(wordmat))
        self.att = Attention(dim_word)
        self.W = nn.Linear(dim_word, dim_word)
        self.V = nn.Linear(dim_word, 1)
        self.output = nn.Linear(dim_hidden * 2, num_class)

    def forward(self, context_ids, context_masks, aspect_ids, aspect_masks):
        context = self.wordemb(context_ids)
        aspect = self.wordemb(aspect_ids)
        context_ = self.att(context, aspect)
        alpha = F.softmax(self.V(torch.tanh(self.W(context) + context_)), 1)  # batch, l, 1
        senti = self.context_enc(context)
        senti = torch.bmm(alpha.transpose(1, 2), senti).squeeze(1)  # batch, 1, dim_hidden * 2
        logit = self.output(senti)
        return logit


class Attention(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.Wc = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.Wa = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.V = nn.Linear(dim_hidden, 1, bias=False)

    def forward(self, context, aspect):
        '''
        :param context: batch, l1, d
        :param aspect: batch, l2, d
        :return: batch, l1, d
        '''
        s = self.Wc(context).unsqueeze(2) + self.Wa(aspect).unsqueeze(1)  # batch, l1, l2, d
        alpha = self.V(torch.tanh(s))  # batch, l1, l2, 1
        alpha = F.softmax(alpha, 2).squeeze(-1)  # batch, l1, l2
        out = torch.bmm(alpha, aspect)  # batch, l1, d
        return out


class ParamGenerator(nn.Module):
    def __init__(self, dim_x, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.Wx = MLP([dim_x, 5, dim_in * dim_out])
        self.Wh = MLP([dim_x, 5, dim_out * dim_out])
        self.Wb = MLP([dim_x, 5, dim_out])

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -0.01, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        '''
        :param x: batch, dim_x
        :return:    Wx batch, dim_in, dim_out
                    Wh batch, dim_out, dim_out
                    B batch, dim_out
        '''
        wx = self.Wx(x).view(-1, self.dim_in, self.dim_out)
        wh = self.Wh(x).view(-1, self.dim_out, self.dim_out)
        wb = self.Wb(x).view(-1, self.dim_out)
        return wx, wh, wb


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        in_sizes = sizes[:-1]
        out_sizes = sizes[1:]
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(in_sizes, out_sizes)])

    def forward(self, x):
        '''
        :param x:
        :return:
        '''
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class BiRNNVP(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.fw_pg = ParamGenerator(dim_word, dim_word, dim_hidden)
        self.bw_pg = ParamGenerator(dim_word, dim_word, dim_hidden)

    def forward(self, x):
        '''
        :param x: batch, l, dim_word
        :return: batch, l, dim_hidden * 2
        '''
        batch, l, _ = x.shape
        fw_list = []
        bw_list = []
        fw_senti = torch.zeros(batch, self.dim_hidden)
        bw_senti = torch.zeros(batch, self.dim_hidden)
        for t in range(l):
            fw_senti = self.step(fw_senti, x[:, t, :], self.fw_pg)
            fw_list.append(fw_senti)
        FW = torch.stack(fw_list, 1)
        for t in range(l - 1, -1, -1):
            bw_senti = self.step(bw_senti, x[:, t, :], self.bw_pg)
            bw_list.append(bw_senti)
        bw_list.reverse()
        BW = torch.stack(bw_list, 1)
        out = torch.cat([FW, BW], -1)
        return out  # batch, l, dim_hidden * 2

    def step(self, last_senti, x, pg):
        '''
        :param last_senti: batch, dim_hidden
        :param x: batch, dim_word
        :param pg:
        :return:
        '''
        Wx, Wh, B = pg(x)
        senti = F.relu(torch.bmm(x.unsqueeze(1), Wx).squeeze(1) + torch.bmm(last_senti.unsqueeze(1), Wh).squeeze(1) + B)
        return senti
