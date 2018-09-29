import xml.etree.ElementTree as ET
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os

padding = '<PADDING>'
unknown = '<UNKNOWN>'


def read_term(fname, wordcounter, charcounter):
    # fout = open(save_fname, 'w')
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    data_list = []
    for sentence in root.findall('sentence'):
        try:
            txt = sentence.find('text').text.lower().rstrip()  # some sentences have space prefix
            aspects = sentence.find('aspectTerms')
            for aspect in aspects.findall('aspectTerm'):
                a = aspect.get('term').lower().strip()
                # if '/' in a:
                #     a = a.split('/')[-1]
                p = aspect.get('polarity')
                f = int(aspect.get('from'))
                t = int(aspect.get('to'))
                left_txt = txt[:f]
                target_txt = txt[f:t]
                if target_txt.lower().strip() != a.lower().strip():
                    raise ValueError('target not same')
                right_txt = txt[t:]
                if p == 'conflict':
                    continue
                p = dic[p]
                sent_tokens = word_tokenize(txt)
                left_tokens = word_tokenize(left_txt)
                right_tokens = word_tokenize(right_txt)
                aspect_tokens = word_tokenize(target_txt)
                for w in sent_tokens:
                    wordcounter[w] += 1
                    for char in list(w):
                        charcounter[char] += 1
                for w in aspect_tokens:
                    wordcounter[w] += 1
                    for char in list(w):
                        charcounter[char] += 1
                data_list.append({
                    'sent': txt,
                    'sent_tokens': sent_tokens,
                    'left': left_txt,
                    'left_tokens': left_tokens,
                    'right': right_txt,
                    'right_tokens': right_tokens,
                    'aspect': target_txt,
                    'aspect_tokens': aspect_tokens,
                    'polarity': p
                })
        except ValueError as e:
            print(e)
            print(txt)
            print(f)
            print(t)
            print(target_txt)
            print(a)
        except Exception:
            pass
    print('get {} instances'.format(len(data_list)))
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


def prepro_term(xml_file_train, xml_file_test, word_limit, sent_limit, aspect_limit, embedding_file):
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
    charcounter = defaultdict(int)
    aspect_cat_train = read_term(xml_file_train, wordcounter, charcounter)
    aspect_cat_test = read_term(xml_file_test, wordcounter, charcounter)
    word2id = create_word2id(wordcounter, word_limit)
    train_data = []
    train_examples = {}
    test_data = []
    test_examples = {}
    for idx, item in tqdm(enumerate(aspect_cat_train)):
        sent_tokens = item['sent_tokens']
        aspect_tokens = item['aspect_tokens']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        aspect_ids = np.zeros([aspect_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(aspect_tokens):
            if i >= aspect_limit:
                break
            aspect_ids[i] = _get_word(w, word2id)
        train_data.append({
            'sent_ids': sent_ids.tolist(),
            'len': min(len(sent_tokens), sent_limit),
            'sent_mask': create_mask(min(len(sent_tokens), sent_limit), sent_limit),
            'aspect_ids': aspect_ids.tolist(),
            'aspect_len': min(len(aspect_tokens), aspect_limit),
            'aspect_mask': create_mask(min(len(aspect_tokens), aspect_limit), aspect_limit),
            'polarity': polarity + 1,
        })
        train_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': item['aspect'],
            'polarity': polarity + 1
        }
    for idx, item in tqdm(enumerate(aspect_cat_test)):
        sent_tokens = item['sent_tokens']
        aspect_tokens = item['aspect_tokens']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        aspect_ids = np.zeros([aspect_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(aspect_tokens):
            if i >= aspect_limit:
                break
            aspect_ids[i] = _get_word(w, word2id)
        k = len(item['left_tokens'])
        m = len(item['aspect_tokens'])
        n = len(item['sent_tokens'])
        test_data.append({
            'sent_ids': sent_ids.tolist(),
            'len': min(len(sent_tokens), sent_limit),
            'sent_mask': create_mask(len(sent_tokens), sent_limit),
            'aspect_ids': aspect_ids.tolist(),
            'aspect_len': min(len(aspect_tokens), aspect_limit),
            'aspect_mask': create_mask(len(aspect_tokens), aspect_limit),
            'polarity': polarity + 1,
        })
        test_examples[str(idx)] = {
            'sent': item['sent'],
            'aspect': item['aspect'],
            'polarity': polarity + 1
        }
    dic = load_embedding(embedding_file, word2id)
    wordmat = load_wordvec(word2id, dic)
    return train_data, train_examples, test_data, test_examples, word2id, wordmat


def create_mask(len, limit):
    a = np.zeros(limit).astype(np.float)
    idx = np.arange(limit) < len
    a[idx] = 1
    return a.tolist()


def save(obj, file, mes):
    print(mes)
    json.dump(obj, open(file, 'w'))


def prepro(config):
    if config.dataset == 'res':
        train_data, train_examples, test_data, test_examples, word2id, wordmat = \
            prepro_term(config.restaurant_train_xml, config.restaurant_test_xml, config.word_limit,
                        config.sent_limit, config.aspect_limit, config.glove_file)
    elif config.dataset == 'lap':
        train_data, train_examples, test_data, test_examples, word2id, wordmat = \
            prepro_term(config.laptop_train_xml, config.laptop_test_xml, config.word_limit,
                        config.sent_limit, config.aspect_limit, config.glove_file)
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
