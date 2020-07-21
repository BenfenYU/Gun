# 统计各类长度句子
# <10：短句  10-15：中长句  15-20：长句  >20：超长句
def jugle_len():
    path = './data/train/a.txt'
    with open(path, 'r', encoding='utf-8') as f:
        len_1, len_2, len_3, len_4 = 0, 0, 0, 0
        for line in f.readlines():
            seq_len = len(line.strip().split(' '))
            if seq_len <= 15:
                len_1 += 1
            elif seq_len <= 20:
                len_2 += 1
            elif seq_len <= 25:
                len_3 += 1
            else:
                len_4 += 1
    return len_1, len_2, len_3, len_4

if __name__ == '__main__':
    print(jugle_len())

