# 提取每个词的依存特征
# 弄成一句一行， 制作词典

def extrac_depen():
    path_1 = 'f:/dataset/train/a_pcfg.txt'
    path_2 = 'f:/dataset/train/dependece.txt'
    with open(path_1, 'r', encoding='utf-8') as f1, open(path_2, 'w', encoding='utf-8') as f2:
        for line in f1.readlines():
            if line == '\n':
                f2.write('\n')
            else:
                f2.write(line.strip().split('(')[0])
                f2.write(' ')

# 利用词典统计总共有多少依存特征
def cal_dep():
    path = 'f:/dataset/train/dependece.txt'
    path_1 = 'f:/dataset/dep_dict.txt'   # 存放依存特征的字典
    dict = {}  # dep2id
    list = []  # 是有序元素，不能消除重复
    with open(path, 'r', encoding='utf-8') as f1, open(path_1, 'w', encoding='utf-8') as f2:
        i = 0
        for line in f1.readlines():
            for word in line.strip().split(' '):
                if word not in dict:
                    dict[word] = i
                    i += 1
        for key in dict:
            f2.write(key)
            f2.write('\n')
    return dict # 存放dep2id的字典

def cal_sentence():
    path_1 = 'f:/dataset/train/a.txt'
    path_2 = 'f:/dataset/train/sen_len.txt'
    with open(path_1, 'r', encoding='utf-8') as f1, open(path_2, 'w', encoding='utf-8') as f2:
        for line in f1.readlines():
            length = len(line.strip().split(' '))
            if length <= 10:
                f2.write('短句')
            elif length <=15:
                f2.write('中长句')
            elif length <= 20:
                f2.write('长句')
            else:
                f2.write('超长句')
            f2.write('\n')
    return


if __name__ == "__main__":
    # f = open('f:/dataset/train/a_pcfg.txt', 'r', encoding='utf-8')
    # print(f.readline().strip().split('('))
    # extrac_depen()
    # dict = cal_dep()
    # print(dict)
    cal_sentence()

