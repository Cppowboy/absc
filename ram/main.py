from ram.preproc import prepro
from ram.ram import RAM
from ram.basic import get_score
from ram.dataset import get_loader
from ram.config import config
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from absl import app
import os
from tqdm import tqdm
import json


def save(sent_ids, lens, aspect_ids, alens, polarity, logits, fout, config):
    def parse_sent(ids, l, dic):
        token_list = []
        for i in range(l):
            if ids[i] in dic:
                token_list.append(dic[ids[i]])
            else:
                token_list.append(dic[0])
        return ' '.join(token_list)

    polarity_list = ['negative', 'neutral', 'positive']
    data_dir = os.path.join(config.model, config.dataset)
    word2id = json.load(open(os.path.join(data_dir, config.word2id_file), 'r'))
    id2word = {v: k for k, v in word2id.items()}
    for sid, l, aid, al, p, logit in zip(sent_ids, lens, aspect_ids, alens, polarity, logits):
        print(parse_sent(sid, l, id2word), file=fout)
        print(parse_sent(aid, al, id2word), file=fout)
        print(polarity_list[p], file=fout)
        print(['{}:{}'.format(t, p) for t, p in zip(polarity_list, logit)], file=fout)


def train(config):
    device = torch.device(config.device)
    # random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    # load data
    data_dir = os.path.join(config.model, config.dataset)
    train_fname = os.path.join(data_dir, config.train_data)
    test_fname = os.path.join(data_dir, config.test_data)
    train_data = get_loader(train_fname, config.batch)
    test_data = get_loader(test_fname, config.batch)
    wordemb = np.loadtxt(os.path.join(data_dir, config.wordmat_file))
    # init model
    model = RAM(dim_word=config.dim_word, dim_hidden=config.dim_hidden, dim_episode=config.dim_episode,
                num_layer=config.num_layer, num_class=config.num_class, wordmat=wordemb,
                dropout_rate=config.dropout_rate, device=device)
    # model = TNet(dim_word=config.dim_word, dim_hidden=config.dim_hidden,
    #              kernel_size=config.kernel_size, num_channel=config.conv_channel,
    #              num_class=config.num_class, cpt_num=config.cpt_num, word_mat=wordemb, dropout_rate=config.dropout_rate,
    #              device=device)
    model = model.to(device)
    # init loss
    cross_entropy = nn.CrossEntropyLoss(weight=config.class_weight)
    # train
    # summary writer
    writer = SummaryWriter('logs/%s/%s/%s' % (config.dataset, config.model, config.timestr))
    model_save_dir = os.path.join(config.model_save, config.dataset, config.model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    result_dir = os.path.join(config.result_save, config.dataset, config.model, 'train')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    best_acc = 0.0
    for epoch in tqdm(range(config.max_epoch)):
        # train
        save_fout = open(os.path.join(result_dir, '{}.txt'.format(epoch)), 'w')
        model.train()
        for i, batch_data in tqdm(enumerate(train_data)):
            model.zero_grad()
            sent_ids, lens, aspect_ids, aspect_lens, polarity, pws = batch_data
            sent_ids, aspect_ids, polarity, pws = sent_ids.to(device), aspect_ids.to(device), polarity.to(
                device), pws.to(device)
            logit = model(sent_ids, aspect_ids, pws)
            save(sent_ids.tolist(), lens.tolist(), aspect_ids.tolist(), aspect_lens.tolist(), polarity.tolist(),
                 logit.tolist(), save_fout, config)
            loss = cross_entropy(logit, polarity)
            writer.add_scalar('loss', loss, len(train_data) * epoch + i)
            loss.backward()
            optim.step()
        # eval
        model.eval()
        # eval on train
        logit_list = []
        rating_list = []
        for batch_data in tqdm(train_data):
            sent_ids, lens, aspect_ids, aspect_lens, polarity, pws = batch_data
            sent_ids, aspect_ids, polarity, pws = sent_ids.to(device), aspect_ids.to(device), polarity.to(
                device), pws.to(device)
            logit = model(sent_ids, aspect_ids, pws)
            # loss = cross_entropy(logit, polarity)
            logit_list.append(logit.cpu().data.numpy())
            rating_list.append(polarity.cpu().data.numpy())
        train_acc, train_precision, train_recall, train_f1 = get_score(np.concatenate(logit_list, 0),
                                                                       np.concatenate(rating_list, 0))
        # writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_precision', train_precision, epoch)
        writer.add_scalar('train_recall', train_recall, epoch)
        writer.add_scalar('train_f1', train_f1, epoch)
        # eval on test
        logit_list = []
        rating_list = []
        for batch_data in tqdm(test_data):
            sent_ids, lens, aspect_ids, aspect_lens, polarity, pws = batch_data
            sent_ids, aspect_ids, polarity, pws = sent_ids.to(device), aspect_ids.to(device), polarity.to(
                device), pws.to(device)
            logit = model(sent_ids, aspect_ids, pws)
            # loss = cross_entropy(logit, polarity)
            logit_list.append(logit.cpu().data.numpy())
            rating_list.append(polarity.cpu().data.numpy())
        test_acc, test_precision, test_recall, test_f1 = get_score(np.concatenate(logit_list, 0),
                                                                   np.concatenate(rating_list, 0))
        # writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_precision', test_precision, epoch)
        writer.add_scalar('test_recall', test_recall, epoch)
        writer.add_scalar('test_f1', test_f1, epoch)
        print('epoch %2d : '
              ' train_acc=%.4f, train_precision=%.4f, train_recall=%.4f,train_f1=%.4f,'
              ' test_acc=%.4f, test_precision=%.4f, test_recall=%.4f, test_f1=%.4f' %
              (epoch, train_acc, train_precision, train_recall, train_f1,
               test_acc, test_precision, test_recall, test_f1))
        # show parameters
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch, bins='doane')
        # save model
        torch.save(model.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(epoch)))
        if test_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best.pth'))
            best_acc = test_acc


def test(config):
    device = torch.device(config.device)
    # load data
    data_dir = os.path.join(config.model, config.dataset)
    test_fname = os.path.join(data_dir, config.test_data)
    test_data = get_loader(test_fname, config.batch)
    wordemb = np.loadtxt(os.path.join(data_dir, config.wordmat_file))
    # charemb = np.loadtxt(os.path.join(data_dir, config.charmat_file))
    # init model
    model = RAM(dim_word=config.dim_word, dim_hidden=config.dim_hidden, dim_episode=config.dim_episode,
                num_layer=config.num_layer, num_class=config.num_class, wordmat=wordemb,
                dropout_rate=config.dropout_rate, device=device)
    # model = TNet(dim_word=config.dim_word, dim_hidden=config.dim_hidden,
    #              kernel_size=config.kernel_size, num_channel=config.conv_channel,
    #              num_class=config.num_class, cpt_num=config.cpt_num, word_mat=wordemb, dropout_rate=config.dropout_rate,
    #              device=device)
    # load model
    model_save_dir = os.path.join(config.model_save, config.dataset, config.model)
    result_dir = os.path.join(config.result_save, config.dataset, config.model, 'test')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_fout = open(os.path.join(result_dir, 'best.txt'), 'w')
    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'best.pth')))
    model = model.to(device)
    model.eval()
    # init loss
    logit_list = []
    rating_list = []
    for batch_data in tqdm(test_data):
        sent_ids, lens, aspect_ids, aspect_lens, polarity, pws = batch_data
        sent_ids, aspect_ids, polarity, pws = sent_ids.to(device), aspect_ids.to(device), polarity.to(
            device), pws.to(device)
        logit = model(sent_ids, aspect_ids, pws)
        save(sent_ids.tolist(), lens.tolist(), aspect_ids.tolist(), aspect_lens.tolist(), polarity.tolist(),
             logit.tolist(), save_fout, config)
        logit_list.append(logit.cpu().data.numpy())
        rating_list.append(polarity.cpu().data.numpy())
    test_acc, test_precision, test_recall, test_f1 = get_score(np.concatenate(logit_list, 0),
                                                               np.concatenate(rating_list, 0))
    print('test_acc=%.4f, test_precision=%.4f, test_recall=%.4f, test_f1=%.4f' %
          (test_acc, test_precision, test_recall, test_f1))


def main(_):
    if config.mode == 'prepro':
        prepro(config)
    elif config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    else:
        raise Exception('unknown mode')


if __name__ == '__main__':
    app.run(main)
