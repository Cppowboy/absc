import torch
import torch.nn as nn
import torch.nn.functional as F
from acsa_gcae.basic import LstmCell, FullConnect, get_mask, masked_softmax
from math import sqrt


# GCAE(gated convolutional network with aspect embedding) for ACSA(aspect category based sentiment analysis)
class ACSA_GCAE(nn.Module):
    def __init__(self, dim_word, num_kernel, num_classification, maxlen, kernel_sizes, dropout_rate, wordemb, targetemb,
                 device):
        super(ACSA_GCAE, self).__init__()
        self.dim_word = dim_word
        self.num_kernel = num_kernel
        self.num_classification = num_classification
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.init_param()
        self.emb_matrix = self.init_emb(wordemb)
        self.target_matrix = self.init_emb(targetemb)
        self.device = device

    def forward(self, sent, target, lens):
        # batch = sent.shape[0]
        x = self.emb_matrix(sent).view(sent.shape[0], sent.shape[1], -1)
        target_x = self.target_matrix(target).view(target.shape[0], -1)
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
        logit = self.fc1(c)
        return logit

    def init_param(self):
        self.convs1 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks) for ks in self.kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks) for ks in self.kernel_sizes])
        self.convs3 = nn.ModuleList([nn.Conv1d(self.dim_word, self.num_kernel, ks, padding=ks - 2) for ks in [3]])
        self.fc1 = FullConnect(len(self.kernel_sizes) * self.num_kernel, self.num_classification)
        self.fc_aspect = FullConnect(self.num_kernel, self.num_kernel)
        self.dropout = nn.Dropout(self.dropout_rate)

    def init_emb(self, embedding):
        num_word, dim_word = embedding.shape
        emb_matrix = nn.Embedding(num_word, dim_word)
        emb_matrix.weight = nn.Parameter(torch.from_numpy(embedding).float())
        return emb_matrix
