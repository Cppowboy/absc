import torch
import torch.nn as nn
import torch.nn.functional as F
from acsa_gcae_g.basic import LstmCell, FullConnect, get_mask, masked_softmax
from math import sqrt


class GCAE(nn.Module):
    def __init__(self, dim_word, num_kernel, num_classification, maxlen, kernel_sizes, dropout_rate, device):
        super(GCAE, self).__init__()
        self.dim_word = dim_word
        self.num_kernel = num_kernel
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.device = device
        self.init_param()

    def forward(self, x, target_x, lens):
        # aspect
        aa = [F.relu(conv(target_x[:, :, None])) for conv in self.convs3]
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        # gates
        a_gate = [F.tanh(conv(x.transpose(1, 2))) for conv in self.convs1]
        s_gate = [F.relu(conv(x.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        c = [i * j for i, j in zip(a_gate, s_gate)]
        # pooling
        c = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in c]
        c = torch.cat(c, 1)
        c = self.dropout(c)
        # logit = self.fc1(c)
        # return logit
        return c

    def init_param(self):
        self.convs1 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks) for ks in self.kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks) for ks in self.kernel_sizes])
        self.convs3 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks, padding=ks - 2) for ks in [3]])
        self.fc_aspect = FullConnect(self.num_kernel, self.num_kernel)
        self.dropout = nn.Dropout(self.dropout_rate)


class ACSA_GCAE_G(nn.Module):
    def __init__(self, dim_word, num_kernel, num_classification, kernel_sizes, maxlen, dropout_rate, wordemb, targetemb,
                 device):
        super(ACSA_GCAE_G, self).__init__()
        self.dim_word = dim_word
        self.num_kernel = num_kernel
        self.num_class = num_classification
        self.kernel_sizes = kernel_sizes
        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
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
        s = self.gcae(words, target, lens)
        s_left = self.left_gcae(left_words, target, left_lens)
        s_right = self.right_gcae(right_words, target, right_lens)
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
        # GCAE module
        self.gcae = GCAE(dim_word=self.dim_word, num_kernel=self.num_kernel, num_classification=self.num_class,
                         maxlen=self.maxlen, dropout_rate=self.dropout_rate, kernel_sizes=self.kernel_sizes,
                         device=self.device)
        self.left_gcae = GCAE(dim_word=self.dim_word, num_kernel=self.num_kernel, num_classification=self.num_class,
                              maxlen=self.maxlen, dropout_rate=self.dropout_rate, kernel_sizes=self.kernel_sizes,
                              device=self.device)
        self.right_gcae = GCAE(dim_word=self.dim_word, num_kernel=self.num_kernel, num_classification=self.num_class,
                               maxlen=self.maxlen, dropout_rate=self.dropout_rate, kernel_sizes=self.kernel_sizes,
                               device=self.device)
        # gate params
        self.w = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.w_left = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.w_right = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.u = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.u_left = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.u_right = nn.Parameter(torch.Tensor(self.dim_word, 1))
        self.b = nn.Parameter(torch.Tensor(1))
        self.b_left = nn.Parameter(torch.Tensor(1))
        self.b_right = nn.Parameter(torch.Tensor(1))
        # logit params
        self.W = nn.Parameter(torch.Tensor(self.dim_word, self.num_class))
        self.B = nn.Parameter(torch.Tensor(self.num_class))
        # init
        ave = 1 / sqrt(self.num_kernel)
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
