def best_key(dic):
    kv = dic.items()
    kv = sorted(kv, key=lambda x: x[1], reverse=True)
    return kv[0][0]


def eval_dic(string):
    string = string.strip()
    string = string[1:-1]
    dic = {}
    for parts in string.split(','):
        parts = parts.strip()[1:-1]
        k, v = parts.split(':')
        v = float(v)
        dic[k] = v
    return dic


if __name__ == '__main__':
    # fin = open('result/lap/model/test/best.txt', 'r')
    # fin = open('result/lap/atae_lstm/test/best.txt', 'r')
    fin = open('result/lap/ram/test/best.txt', 'r')
    lines = fin.readlines()
    n = len(lines)
    total = 0
    correct = 0
    for i in range(n // 4):
        total += 1
        context = lines[i * 4].strip()
        aspect = lines[i * 4 + 1].strip()
        polarity = lines[i * 4 + 2].strip()
        predict = lines[i * 4 + 3].strip()
        predict_dict = eval_dic(predict)
        predict = best_key(predict_dict)
        if predict == polarity:
            correct += 1
        else:
            print('===' * 30)
            print(context)
            print(aspect)
            print(polarity)
            print(predict_dict)
    print('total {}, correct {}'.format(total, correct))
