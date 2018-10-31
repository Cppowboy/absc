import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle

# from transformer.analysis import rocstories as rocstories_analysis
from transformer_sep.datasets import semeval
from transformer_sep.model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from transformer_sep.opt import OpenAIAdam
from transformer_sep.text_utils import TextEncoder
from transformer_sep.utils import (encode_dataset, iter_data,
                                   ResultLogger, make_path)
from transformer_sep.loss import MultipleChoiceLossCompute, ClassificationLossCompute


def transform_semeval(X1, X2):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, n_ctx, 2), dtype=np.int32)
    amb = np.zeros((n_batch, n_asp, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, n_ctx), dtype=np.float32)
    ammb = np.zeros((n_batch, n_asp), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2), in enumerate(zip(X1, X2)):
        x12 = [start] + x1[:max_len] + [clf_token]
        x22 = [start] + x2[:asp_max_len] + [clf_token]
        l12 = len(x12)
        l22 = len(x22)
        xmb[i, :l12, 0] = x12
        mmb[i, :l12] = 1
        amb[i, :l22, 0] = x22
        ammb[i, :l22] = 1
    # Position information that is added to the input embeddings in the TransformerModel
    xmb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    amb[:, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_asp)
    return xmb, amb, mmb, ammb


def iter_apply(Xs1, Xs2, Ms1, Ms2, Ys):
    # fns = [lambda x: np.concatenate(x, 0), lambda x: float(np.sum(x))]
    logits = []
    cost = 0
    with torch.no_grad():
        dh_model.eval()
        for xmb1, xmb2, mmb1, mmb2, ymb in iter_data(Xs1, Xs2, Ms1, Ms2, Ys, n_batch=n_batch_train, truncate=False,
                                                     verbose=True):
            n = len(xmb1)
            XMB1 = torch.tensor(xmb1, dtype=torch.long).to(device)
            XMB2 = torch.tensor(xmb2, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB1 = torch.tensor(mmb1).to(device)
            MMB2 = torch.tensor(mmb2).to(device)
            _, clf_logits = dh_model(XMB1, XMB2)
            clf_logits *= n
            clf_losses = compute_loss_fct(XMB1, XMB2, YMB, MMB1, MMB2, clf_logits, only_return_losses=True)
            clf_losses *= n
            logits.append(clf_logits.to("cpu").numpy())
            cost += clf_losses.sum().item()
        logits = np.concatenate(logits, 0)
    return logits, cost


def iter_predict(Xs1, Xs2, Ms1, Ms2):
    logits = []
    with torch.no_grad():
        dh_model.eval()
        for xmb1, xmb2, mmb1, mmb2 in iter_data(Xs1, Xs2, Ms1, Ms2, n_batch=n_batch_train, truncate=False,
                                                verbose=True):
            n = len(xmb1)
            XMB1 = torch.tensor(xmb1, dtype=torch.long).to(device)
            XMB2 = torch.tensor(xmb2, dtype=torch.long).to(device)
            MMB1 = torch.tensor(mmb1).to(device)
            MMB2 = torch.tensor(mmb2).to(device)
            _, clf_logits = dh_model(XMB1, XMB2)
            logits.append(clf_logits.to("cpu").numpy())
    logits = np.concatenate(logits, 0)
    return logits


def log(save_dir, desc):
    global best_score
    print("Logging")
    tr_logits, tr_cost = iter_apply(trX1[:n_valid], trX2[:n_valid], trM1[:n_valid], trM2[:n_valid], trY[:n_valid])
    va_logits, va_cost = iter_apply(vaX1, vaX2, vaM1, vaM2, vaY)
    tr_cost = tr_cost / len(trY[:n_valid])
    va_cost = va_cost / n_valid
    tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc, va_acc=va_acc)
    print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))
    print('Train:')
    print(precision_recall_fscore_support(trY[:n_valid], np.argmax(tr_logits, 1), average='macro'))
    print('Val:')
    print(precision_recall_fscore_support(vaY, np.argmax(va_logits, 1), average='macro'))


def run_epoch():
    for xmb1, xmb2, mmb1, mmb2, ymb in iter_data(*shuffle(trX1, trX2, trM1, trM2, trYt, random_state=np.random),
                                                 n_batch=n_batch_train, truncate=True, verbose=True):
        global n_updates
        dh_model.train()
        XMB1 = torch.tensor(xmb1, dtype=torch.long).to(device)
        XMB2 = torch.tensor(xmb2, dtype=torch.long).to(device)
        YMB = torch.tensor(ymb, dtype=torch.long).to(device)
        MMB1 = torch.tensor(mmb1).to(device)
        MMB2 = torch.tensor(mmb2).to(device)
        lm_logits, clf_logits = dh_model(XMB1, XMB2)
        compute_loss_fct(XMB1, XMB2, YMB, MMB1, MMB2, clf_logits, lm_logits)
        n_updates += 1
        if n_updates in [1000, 2000, 4000, 8000, 16000, 32000] and n_epochs == 0:
            log(save_dir, desc)


argmax = lambda x: np.argmax(x, 1)

pred_fns = {
    'rocstories': argmax,
}

filenames = {
    'rocstories': 'ROCStories.tsv',
}

label_decoders = {
    'rocstories': None,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/restaurant')
    # parser.add_argument('--data_dir', type=str, default='data/laptop')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=120)
    parser.add_argument('--n_asp', type=int, default=10)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='transformer/model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='transformer/model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Constants
    dataset = args.dataset
    n_ctx = args.n_ctx
    n_asp = args.n_asp
    save_dir = args.save_dir
    desc = args.desc
    data_dir = args.data_dir
    log_dir = args.log_dir
    submission_dir = args.submission_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)
    text_encoder = TextEncoder(args.encoder_path, args.bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    print("Encoding dataset...")
    aaaa = semeval(data_dir)
    ((trX1, trX2, trY),
     (vaX1, vaX2, vaY)) = encode_dataset(*aaaa, encoder=text_encoder)
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_special = 3
    max_len = n_ctx // 2 - 2
    asp_max_len = n_asp // 2 - 2
    n_ctx = min(max(
        [len(x1[:max_len]) + len(x2[:max_len]) for x1, x2 in zip(trX1, trX2)]
        + [len(x1[:max_len]) + len(x2[:max_len]) for x1, x2 in zip(vaX1, vaX2)]
    ) + 3, n_ctx)
    vocab = n_vocab + n_special + n_ctx
    trX1, trX2, trM1, trM2 = transform_semeval(trX1, trX2)
    vaX1, vaX2, vaM1, vaM2 = transform_semeval(vaX1, vaX2)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = args.n_batch * max(n_gpu, 1)
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = DoubleHeadModel(args, clf_token, 'inference', vocab, n_ctx, n_asp)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)
    compute_loss_fct = ClassificationLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special, path='transformer/model/')
    load_openai_pretrained_model(dh_model.asp_transformer, n_ctx=n_asp, n_special=n_special, path='transformer/model/')

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    n_updates = 0
    n_epochs = 0
    if dataset != 'stsb':
        trYt = trY

    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1
        log(save_dir, desc)
