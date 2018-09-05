import xml.etree.ElementTree as ET
from nltk import word_tokenize
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os

padding = '<PADDING>'
unknown = '<UNKNOWN>'


def read_seperate(fname, wordcounter, targetcounter):
    # fout = open(save_fname, 'w')
    dic = {'positive': 1, 'neutral': 0, 'negative': -1}
    tree = ET.parse(fname)
    root = tree.getroot()
    data_list = []
    for sentence in root.findall('sentence'):
        try:
            txt = sentence.find('text').text.lower().rstrip()
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
                if target_txt != a:
                    raise Exception('target not same')
                right_txt = txt[t:]
                if p == 'conflict':
                    continue
                p = dic[p]
                left_words, target_words, right_words = word_tokenize(left_txt), \
                                                        word_tokenize(target_txt), word_tokenize(right_txt)
                processed_txt = ' '.join(left_words + target_words + right_words)
                # sent_tokens = word_tokenize(processed_txt)
                sent_tokens = left_words + target_words + right_words
                for w in sent_tokens:
                    wordcounter[w] += 1
                targetcounter[a] += 1
                # m = len(left_words)
                # n = len(target_words)
                # print('%s ||| %d %d ||| %d' % (processed_txt, m, m + n - 1, p), file=fout)
                data_list.append({
                    'sent': processed_txt,
                    'sent_tokens': sent_tokens,
                    'left': left_txt,
                    'left_tokens': left_words,
                    'right': right_txt,
                    'right_tokens': right_words,
                    'aspect': a,
                    'polarity': p
                })
        except:
            pass
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
    emb_file = open(emb_fname, 'r')
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


def prepro_seperate(xml_file_train, xml_file_test, word_limit, sent_limit, embedding_file):
    wordcounter = defaultdict(int)
    targetcounter = defaultdict(int)
    aspect_train = read_seperate(xml_file_train, wordcounter, targetcounter)
    aspect_test = read_seperate(xml_file_test, wordcounter, targetcounter)
    word2id = create_word2id(wordcounter, word_limit)
    target2id = create_word2id(targetcounter, word_limit)
    train_data = []
    train_examples = {}
    test_data = []
    test_examples = {}
    for idx, item in tqdm(enumerate(aspect_train)):
        sent_tokens = item['sent_tokens']
        left_tokens = item['left_tokens']
        right_tokens = item['right_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        left_ids = np.zeros([sent_limit])
        right_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(left_tokens):
            if i >= sent_limit:
                break
            left_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(right_tokens):
            if i >= sent_limit:
                break
            right_ids[i] = _get_word(w, word2id)
        train_data.append({
            'sent_ids': sent_ids.tolist(),
            'left_ids': left_ids.tolist(),
            'right_ids': right_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit),
            'left_len': min(len(left_tokens), sent_limit),
            'right_len': min(len(right_tokens), sent_limit)
        })
        train_examples[str(idx)] = {
            'sent': item['sent'],
            'left': item['left'],
            'right': item['right'],
            'aspect': aspect,
            'polarity': polarity + 1
        }
    for idx, item in enumerate(aspect_test):
        sent_tokens = item['sent_tokens']
        left_tokens = item['left_tokens']
        right_tokens = item['right_tokens']
        aspect = item['aspect']
        polarity = item['polarity']
        sent_ids = np.zeros([sent_limit])
        left_ids = np.zeros([sent_limit])
        right_ids = np.zeros([sent_limit])
        for i, w in enumerate(sent_tokens):
            if i >= sent_limit:
                break
            sent_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(left_tokens):
            if i >= sent_limit:
                break
            left_ids[i] = _get_word(w, word2id)
        for i, w in enumerate(right_tokens):
            if i >= sent_limit:
                break
            right_ids[i] = _get_word(w, word2id)
        test_data.append({
            'sent_ids': sent_ids.tolist(),
            'left_ids': left_ids.tolist(),
            'right_ids': right_ids.tolist(),
            'aspect_id': _get_word(aspect, target2id),
            'polarity': polarity + 1,
            'len': min(len(sent_tokens), sent_limit),
            'left_len': min(len(left_tokens), sent_limit),
            'right_len': min(len(right_tokens), sent_limit)
        })
        test_examples[str(idx)] = {
            'sent': item['sent'],
            'left': item['left'],
            'right': item['right'],
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
    if config.dataset == 'res':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_seperate(
            config.restaurant_train_xml,
            config.restaurant_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
    elif config.dataset == 'lap':
        train_data, train_examples, test_data, test_examples, word2id, target2id, wordmat, targetmat = prepro_seperate(
            config.laptop_train_xml,
            config.laptop_test_xml,
            config.word_limit,
            config.sent_limit,
            config.glove_file)
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
