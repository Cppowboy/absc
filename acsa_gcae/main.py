from acsa_gcae.preproc import prepro
from acsa_gcae.acsa_gcae import ACSA_GCAE
from acsa_gcae.basic import get_score
from acsa_gcae.dataset import get_loader
from acsa_gcae.config import config
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from absl import app
import os
from tqdm import tqdm


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
    targetemb = np.loadtxt(os.path.join(data_dir, config.targetmat_file))
    # init model
    model = ACSA_GCAE(dim_word=config.dim_word, num_kernel=config.num_kernel, num_classification=config.num_class,
                      maxlen=config.sent_limit, kernel_sizes=config.kernel_sizes, dropout_rate=config.dropout_rate,
                      wordemb=wordemb, targetemb=targetemb, device=device)
    model = model.to(device)
    # init loss
    cross_entropy = nn.CrossEntropyLoss()
    # train
    # summary writer
    writer = SummaryWriter('logs/%s/%s/%s' % (config.dataset, config.model, config.timestr))
    model_save_dir = os.path.join(config.model_save, config.dataset, config.model)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    optim = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    best_acc = 0.0
    for epoch in tqdm(range(config.max_epoch)):
        # train
        model.train()
        for batch_data in tqdm(train_data):
            model.zero_grad()
            sent_ids, aspect_id, polarity, lens = batch_data
            sent_ids, aspect_id, polarity, lens = sent_ids.to(device), aspect_id.to(device), \
                                                  polarity.to(device), lens.to(device)
            logit = model(sent_ids, aspect_id, lens)
            loss = cross_entropy(logit, polarity)
            loss.backward()
            optim.step()
        # eval
        model.eval()
        # eval on train
        logit_list = []
        rating_list = []
        for batch_data in tqdm(train_data):
            sent_ids, aspect_id, polarity, lens = batch_data
            sent_ids, aspect_id, polarity, lens = sent_ids.to(device), aspect_id.to(device), \
                                                  polarity.to(device), lens.to(device)
            logit = model(sent_ids, aspect_id, lens)
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
            sent_ids, aspect_id, polarity, lens = batch_data
            sent_ids, aspect_id, polarity, lens = sent_ids.to(device), aspect_id.to(device), \
                                                  polarity.to(device), lens.to(device)
            logit = model(sent_ids, aspect_id, lens)
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
    targetemb = np.loadtxt(os.path.join(data_dir, config.targetmat_file))
    # init model
    model = ACSA_GCAE(dim_word=config.dim_word, num_kernel=config.num_kernel, num_classification=config.num_class,
                      maxlen=config.sent_limit, kernel_sizes=config.kernel_sizes, dropout_rate=config.dropout_rate,
                      wordemb=wordemb, targetemb=targetemb, device=device)
    # load model
    model_save_dir = os.path.join(config.model_save, config.dataset, config.model)
    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'best.pth')))
    model = model.to(device)
    model.eval()
    # init loss
    logit_list = []
    rating_list = []
    for batch_data in tqdm(test_data):
        sent_ids, aspect_id, polarity, lens = batch_data
        sent_ids, aspect_id, polarity, lens = sent_ids.to(device), aspect_id.to(device), \
                                              polarity.to(device), lens.to(device)
        logit = model(sent_ids, aspect_id, lens)
        logit_list.append(logit.cpu().data.numpy())
        rating_list.append(polarity.cpu().data.numpy())
    test_acc, test_precision, test_recall, test_f1 = get_score(np.concatenate(logit_list, 0),
                                                               np.concatenate(rating_list, 0))
    print(' test_acc=%.4f, test_precision=%.4f, test_recall=%.4f, test_f1=%.4f' %
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
