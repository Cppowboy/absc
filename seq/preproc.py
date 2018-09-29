import xml.etree.ElementTree as ET
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os

padding = '<PADDING>'
unknown = '<UNKNOWN>'


def read_term(fname, wordcounter):
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    data_list = []
    for sentence in root.findall('sentence'):
        try:
            txt = sentence.find('text').text.lower().strip()
            sent_tokens = word_tokenize(txt)
            for w in sent_tokens:
                wordcounter[w] += 1
            aspects = sentence.find('aspectTerms')
            aspect_list = []
            for aspect in aspects.findall('aspectTerm'):
                a = aspect.get('term').lower().rstrip()
                p = aspect.get('polarity')
                f = int(aspect.get('from'))
                t = int(aspect.get('to'))
                left_txt = txt[:f]
                target_txt = txt[f:t]
                if target_txt != a:
                    raise Exception('target not same')
                right_txt = txt[t:]
                if p == 'conflict':
                    continue
                p = dic[p]
                left_tokens = word_tokenize(left_txt)
                right_tokens = word_tokenize(right_txt)
                aspect_tokens = word_tokenize(target_txt)
                aspect_list.append({
                    'aspect': target_txt,
                    'aspect_tokens': aspect_tokens,
                    'aspect_start': len(left_tokens),
                    'aspect_end': len(left_tokens) + len(aspect_tokens),
                    'polarity': p
                })
            data_list.append({
                'sent': txt,
                'sent_tokens': sent_tokens,
                'aspect_list': aspect_list
            })
        except Exception as e:
            # print(e)
            pass
    return data_list


def load_embedding(emb_fname, word2id):
    '''
    read glove embedding file, return word embeddings for words in word2id
    :param emb_fname:
    :param word2id:
    :return:
    '''
    print('loading word embedding file {}'.format(emb_fname))
    dic = {}
    emb_file = open(emb_fname, 'r', encoding='utf-8')
    for line in tqdm(emb_file):
        try:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = list(map(float, parts[1:]))
            if word in word2id:
                dic[word] = np.array(vec)
        except:
            pass
    print('load %d words' % (len(dic)))
    return dic


def load_wordvec(wordlist, emb_dic):
    '''
    :param wordlist:
    :param emb_dic:
    :return: np.array of the wordmat
    '''
    print('building wordmat')
    num_word = len(wordlist) + 2
    dim_word = len(list(emb_dic.values())[0])
    # embedding = np.random.uniform(-0.25, 0.25, [num_word, dim_word])
    embedding = np.zeros([num_word, dim_word])
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


def prepro_term(xml_file_train, xml_file_test, word_limit, sent_limit, embedding_file):
    '''
    tnet prepro
    :param xml_file_train:
    :param xml_file_test:
    :param word_limit:
    :param sent_limit:
    :param embedding_file:
    :return:
    '''
    wordcounter = defaultdict(int)
    aspect_cat_train = read_term(xml_file_train, wordcounter)
    aspect_cat_test = read_term(xml_file_test, wordcounter)
    word2id = create_word2id(wordcounter, word_limit)
    train_data = []
    train_examples = {}
    test_data = []
    test_examples = {}
    for idx, item in tqdm(enumerate(aspect_cat_train)):
        sent_tokens = item['sent_tokens']
        sent_ids = np.zeros([sent_limit])
        context_labels = np.zeros([sent_limit])
        aspect_mask = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, aspect in enumerate(item['aspect_list']):
            s = aspect['aspect_start']
            e = aspect['aspect_end']
            polarity = aspect['polarity']
            context_labels[s:e] = polarity
            aspect_mask[s:e] = 1.0
        for i, aspect in enumerate(item['aspect_list']):
            aspect_pos = np.zeros([sent_limit])
            a = aspect['aspect']
            s = aspect['aspect_start']
            e = aspect['aspect_end']
            polarity = aspect['polarity']
            aspect_pos[s:e] = 1.0
            train_data.append({
                'context_ids': sent_ids.tolist(),
                'context_label': (context_labels + 1).tolist(),
                'aspect_mask': aspect_mask.tolist(),
                'aspect_pos': aspect_pos.tolist(),
                'polarity': polarity + 1,
            })
            train_examples[str(idx)] = {
                'sent': item['sent'],
                'aspect': a,
                'polarity': polarity + 1
            }
    for idx, item in tqdm(enumerate(aspect_cat_test)):
        sent_tokens = item['sent_tokens']
        sent_ids = np.zeros([sent_limit])
        context_labels = np.zeros([sent_limit])
        aspect_mask = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, aspect in enumerate(item['aspect_list']):
            s = aspect['aspect_start']
            e = aspect['aspect_end']
            polarity = aspect['polarity']
            context_labels[s:e] = polarity
            aspect_mask[s:e] = 1.0
        for i, aspect in enumerate(item['aspect_list']):
            aspect_pos = np.zeros([sent_limit])
            a = aspect['aspect']
            s = aspect['aspect_start']
            e = aspect['aspect_end']
            polarity = aspect['polarity']
            aspect_pos[s:e] = 1.0
            test_data.append({
                'context_ids': sent_ids.tolist(),
                'context_label': (context_labels + 1).tolist(),
                'aspect_mask': aspect_mask.tolist(),
                'aspect_pos': aspect_pos.tolist(),
                'polarity': polarity + 1,
            })
            test_examples[str(idx)] = {
                'sent': item['sent'],
                'aspect': a,
                'polarity': polarity + 1
            }
    dic = load_embedding(embedding_file, word2id)
    wordmat = load_wordvec(word2id, dic)
    return train_data, train_examples, test_data, test_examples, word2id, wordmat


def save(obj, file, mes):
    print(mes)
    json.dump(obj, open(file, 'w'))


def prepro(config):
    if config.dataset == 'res':
        train_data, train_examples, test_data, test_examples, word2id, wordmat = \
            prepro_term(config.restaurant_train_xml, config.restaurant_test_xml, config.word_limit,
                        config.sent_limit, config.glove_file)
    elif config.dataset == 'lap':
        train_data, train_examples, test_data, test_examples, word2id, wordmat = \
            prepro_term(config.laptop_train_xml, config.laptop_test_xml, config.word_limit,
                        config.sent_limit, config.glove_file)
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
    np.savetxt(os.path.join(data_dir, config.wordmat_file), wordmat)
