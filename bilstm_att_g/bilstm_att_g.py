import torch
import torch.nn as nn
import torch.nn.functional as F
from bilstm_att_g.basic import LstmCell, get_mask, masked_softmax
from math import sqrt


class Attention(nn.Module):
    def __init__(self, dim_word, dim_hidden, dim_att_hidden, device):
        super(Attention, self).__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.dim_att_hidden = dim_att_hidden
        self.device = device
        self.init_param()

    def forward(self, words, target, lens):
        batch, maxlen, _ = words.shape
        h, c = torch.zeros(batch, self.dim_hidden).to(self.device), torch.zeros(batch, self.dim_hidden).to(self.device)
        h_list = []
        mask = get_mask(maxlen, lens)
        for t in range(maxlen):
            _h, _c = self.cell(words[:, t, :], h, c)
            m = mask[:, t]
            h = m[:, None] * _h + (1 - m)[:, None] * h
            c = m[:, None] * _c + (1 - m)[:, None] * c
            h_list.append(h)
        H = torch.stack(h_list, dim=1)
        # attention
        matrix_aspect = torch.zeros(batch, maxlen, self.dim_word).to(self.device) + target[:, None, :]
        hhhh = torch.cat([H, matrix_aspect], dim=-1)
        M_tmp = F.tanh(torch.matmul(hhhh, self.w1) + self.b1)
        alpha_tmp = masked_softmax(torch.matmul(M_tmp, self.w), mask)

        s = torch.bmm(alpha_tmp[:, None, :], H).squeeze()
        return s

    def init_param(self):
        self.cell = LstmCell(input_size=self.dim_word, hidden_size=self.dim_hidden)
        dim_param = self.dim_word + self.dim_hidden
        self.w1 = nn.Parameter(torch.Tensor(dim_param, self.dim_att_hidden))
        self.b1 = nn.Parameter(torch.Tensor(self.dim_att_hidden))
        self.w = nn.Parameter(torch.Tensor(self.dim_att_hidden, ))
        u = 1 / sqrt(self.dim_hidden)
        nn.init.uniform_(self.w1, -u, u)
        nn.init.uniform_(self.w, -u, u)
        self.b1.data.zero_()


class ABSA_Bilstm_Att_G(nn.Module):
    def __init__(self, dim_word, dim_hidden, dim_att_hidden, num_classification, wordemb, targetemb, device):
        super(ABSA_Bilstm_Att_G, self).__init__()
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.dim_att_hidden = dim_att_hidden
        self.num_class = num_classification
        self.wordemb = self.init_emb(wordemb)
        self.targetemb = self.init_emb(targetemb)
        self.device = device
        self.init_param()

    def forward(self, left_words, right_words, words, target, left_lens, right_lens, lens):
        batch, maxlen = words.shape
        left_words = self.wordemb(left_words).view(batch, maxlen, self.dim_word)
        right_words = self.wordemb(right_words).view(batch, maxlen, self.dim_word)
        words = self.wordemb(words).view(batch, maxlen, self.dim_word)
        target = self.targetemb(target).view(batch, self.dim_word)
        s = self.attention(words, target, lens)
        s_left = self.left_attention(left_words, target, left_lens)
        s_right = self.right_attention(right_words, target, right_lens)
        z = torch.matmul(s, self.w) + torch.matmul(target, self.u) + self.b
        z_left = torch.matmul(s_left, self.w_left) + torch.matmul(target, self.u_left) + self.b_left
        z_right = torch.matmul(s_right, self.w_right) + torch.matmul(target, self.u_right) + self.b_right
        z, z_left, z_right = z.exp(), z_left.exp(), z_right.exp()
        gate_sum = z + z_left + z_right
        z, z_left, z_right = z / gate_sum, z_left / gate_sum, z_right / gate_sum
        final_s = z * s + z_left * s_left + z_right * s_right
        logit = torch.matmul(final_s, self.W) + self.B
        return logit

    def init_param(self):
        # attention module
        self.attention = Attention(dim_word=self.dim_word, dim_hidden=self.dim_hidden,
                                   dim_att_hidden=self.dim_att_hidden, device=self.device)

        self.left_attention = Attention(dim_word=self.dim_word, dim_hidden=self.dim_hidden,
                                        dim_att_hidden=self.dim_att_hidden, device=self.device)
        self.right_attention = Attention(dim_word=self.dim_word, dim_hidden=self.dim_hidden,
                                         dim_att_hidden=self.dim_att_hidden, device=self.device)
        # gate params
        self.w = nn.Parameter(torch.Tensor(self.dim_hidden, 1))
        self.w_left = nn.Parameter(torch.Tensor(self.dim_hidden, 1))
        self.w_right = nn.Parameter(torch.Tensor(self.dim_hidden, 1))
        self.u = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.u_left = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.u_right = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.b = nn.Parameter(torch.Tensor(1))
        self.b_left = nn.Parameter(torch.Tensor(1))
        self.b_right = nn.Parameter(torch.Tensor(1))
        # logit params
        self.W = nn.Parameter(torch.Tensor(self.dim_hidden, self.num_class))
        self.B = nn.Parameter(torch.Tensor(self.num_class))
        # init
        ave = 1 / sqrt(self.dim_hidden)
        nn.init.uniform_(self.w, -ave, ave)
        nn.init.uniform_(self.u, -ave, ave)
        nn.init.uniform_(self.w_left, -ave, ave)
        nn.init.uniform_(self.u_left, -ave, ave)
        nn.init.uniform_(self.w_right, -ave, ave)
        nn.init.uniform_(self.u_right, -ave, ave)
        nn.init.uniform_(self.W, -ave, ave)
        self.b.data.zero_()
        self.b_left.data.zero_()
        self.b_right.data.zero_()
        self.B.data.zero_()

    def init_emb(self, embedding):
        num_word, dim_word = embedding.shape
        emb_matrix = nn.Embedding(num_word, dim_word)
        emb_matrix.weight = nn.Parameter(torch.from_numpy(embedding).float())
        return emb_matrix
