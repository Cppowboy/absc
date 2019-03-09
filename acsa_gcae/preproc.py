import xml.etree.ElementTree as ET
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os

padding = '<PADDING>'
unknown = '<UNKNOWN>'


def read_category(fname, wordcounter, targetcounter):
    '''
    read aspect category data from xml file
    :param fname:
    :param wordcounter:
    :param targetcounter:
    :return:
    '''
    print('reading aspect category from {}'.format(fname))
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    data_list = []
    for sentence in tqdm(root.findall('sentence')):
        txt = sentence.find('text').text.lower().rstrip()
        words = word_tokenize(txt)
        aspects = sentence.find('aspectCategories')
        for aspect in aspects.findall('aspectCategory'):
            a = aspect.get('category').lower().strip()
            # if '/' in a:
            #     a = a.split('/')[-1]
            p = aspect.get('polarity')
            if p == 'conflict':
                continue
            p = dic[p]
            for w in words:
                wordcounter[w] += 1
            targetcounter[a] += 1
            data_list.append({
                'sent': txt,
                'sent_tokens': words,
                'aspect': a,
                'polarity': p
            })
    return data_list


def read_term(fname, wordcounter, targetcounter):
    '''
    read aspect term data from xml file
    :param fname:
    :param wordcounter:
    :param targetcounter:
    :return:
    '''
    print('reading aspect term from {}'.format(fname))
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    bad_sent = 0
    data_list = []
    for sentence in tqdm(root.findall('sentence')):
        try:
            txt = sentence.find('text').text.lower().rstrip()
            words = word_tokenize(txt)
            aspects = sentence.find('aspectTerms')
            for aspect in aspects.findall('aspectTerm'):
                a = aspect.get('term').lower().strip()
                # if '/' in a:
                #     a = a.split('/')[-1]
                p = aspect.get('polarity')
                if p == 'conflict':
                    continue
                p = dic[p]
                for w in words:
                    wordcounter[w] += 1
                targetcounter[a] += 1
                data_list.append({
                    'sent': txt,
                    'sent_tokens': words,
                    'aspect': a,
                    'polarity': p
                })
        except:
            bad_sent += 1
    print('bad sent %d' % bad_sent)
    return data_list


def load_embedding(emb_fname, word2id, target2id):
    '''
    read glove embedding file, return word embeddings for words in word2id
    :param emb_fname:
    :param word2id:
    :return:
    '''
    print('loading word embedding file {}'.format(emb_fname))
    dic = {}
    target_dic = {}
    emb_file = open(emb_fname, 'r', encoding='utf-8')
    for line in tqdm(emb_file):
        try:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = list(map(float, parts[1:]))
            if word in word2id:
                dic[word] = np.array(vec)
            if word in target2id:
                target_dic[word] = np.array(vec)
        except:
            pass
    print('load %d words, %d target words' % (len(dic), len(target_dic)))
    return dic, target_dic


def load_wordvec(wordlist, emb_dic):
    '''
    :param wordlist:
    :param emb_dic:
    :return: np.array of the wordmat
    '''
    print('building wordmat')
    num_word = len(wordlist) + 1
    dim_word = len(list(emb_dic.values())[0])
    embedding = np.random.normal(0, 0.01, [num_word, dim_word])
    not_found = 0
    for word, idx in wordlist.items():
        try:
            embedding[idx] = emb_dic[word]
        except:
            not_found += 1
    print('%d words not found' % not_found)
    return embedding


def create_word2id(counter, limit):
    word2id = {}
    word2id[padding] = 0
    word2id[unknown] = 1
    for idx, w in enumerate(counter.keys(), 2):
        if counter[w] > limit:
            word2id[w] = idx
    return word2id


def _get_word(token, word2id):
    if token in word2id:
        return word2id[token]
    else:
        return word2id[unknown]


def prepro_cat(xml_file_train, xml_file_test, word_limit, sent_limit, embedding_file):
    wordcounter = defaultdict(int)
    targetcounter = defaultdict(int)
    aspect_cat_train = read_category(xml_file_train, wordcounter, targetcounter)
    aspect_cat_test = read_category(xml_file_test, wordcounter, targetcounter)
    word2id = create_word2id(wordcounter, word_limit)
    target2id = create_word2id(targetcounter, word_limit)
    train_data = []
    train_examples = {}
    test_data = []
    test_examples = {}
    for idx, item in tqdm(enumerate(aspect_cat_train)):
        sent_tokens = item['sent_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        train_data.append({
            'sent_ids': sent_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit)
        })
        train_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': aspect,
            'polarity': polarity + 1
        }
    for idx, item in enumerate(aspect_cat_test):
        sent_tokens = item['sent_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        test_data.append({
            'sent_ids': sent_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit)
        })
        test_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': aspect,
            'polarity': polarity + 1
        }
    dic, target_dic = load_embedding(embedding_file, word2id, target2id)
    wordmat = load_wordvec(word2id, dic)
    targetmat = load_wordvec(target2id, target_dic)
    return train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat


def prepro_term(xml_file_train, xml_file_test, word_limit, sent_limit, embedding_file):
    wordcounter = defaultdict(int)
    targetcounter = defaultdict(int)
    aspect_cat_train = read_term(xml_file_train, wordcounter, targetcounter)
    aspect_cat_test = read_term(xml_file_test, wordcounter, targetcounter)
    word2id = create_word2id(wordcounter, word_limit)
    target2id = create_word2id(targetcounter, word_limit)
    train_data = []
    train_examples = {}
    test_data = []
    test_examples = {}
    for idx, item in tqdm(enumerate(aspect_cat_train)):
        sent_tokens = item['sent_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        train_data.append({
            'sent_ids': sent_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit)
        })
        train_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': aspect,
            'polarity': polarity + 1
        }
    for idx, item in enumerate(aspect_cat_test):
        sent_tokens = item['sent_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        test_data.append({
            'sent_ids': sent_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit)
        })
        test_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': aspect,
            'polarity': polarity + 1
        }
    dic, target_dic = load_embedding(embedding_file, word2id, target2id)
    wordmat = load_wordvec(word2id, dic)
    targetmat = load_wordvec(target2id, target_dic)
    return train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat


def save(obj, file, mes):
    print(mes)
    json.dump(obj, open(file, 'w'))


def prepro(config):
    if config.dataset == 'res_cat':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_cat(
            config.restaurant_train_xml,
            config.restaurant_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
    elif config.dataset == 'stock':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_cat(
            config.stock_train_xml,
            config.stock_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
    elif config.dataset == 'res':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_term(
            config.restaurant_train_xml,
            config.restaurant_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
    elif config.dataset == 'lap':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_term(
            config.laptop_train_xml,
            config.laptop_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
    elif config.dataset == 'twitter':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_term(
            config.twitter_train_xml,
            config.twitter_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file
            )
    else:
        raise Exception('unknown dataset')
    data_dir = os.path.join(config.model, config.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print('saving to {}'.format(data_dir))
    save(train_data, os.path.join(data_dir, config.train_data), 'saving train data')
    save(train_examples, os.path.join(data_dir, config.train_examples), 'saving train examples')
    save(test_data, os.path.join(data_dir, config.test_data), 'saving test data')
    save(test_examples, os.path.join(data_dir, config.test_examples), 'saving test examples')
    save(word2id, os.path.join(data_dir, config.word2id_file), 'saving word2id file')
    save(target2id, os.path.join(data_dir, config.target2id_file), 'saving target2id file')
    np.savetxt(os.path.join(data_dir, config.wordmat_file), wordmat)
    np.savetxt(os.path.join(data_dir, config.targetmat_file), targetmat)
