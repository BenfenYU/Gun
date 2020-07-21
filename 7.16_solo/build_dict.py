# return word2id, pos2id, tag2id
def Build_dict():
    '''读取的是三个txt文件，返回字典格式'''
    word_path = './data/dict/word_dict.txt'
    pos_path = './data/dict/pos_dict.txt'
    tag_path = './data/dict/tag_dict.txt'
    word2id, pos2id, tag2id, id2tag = {}, {}, {}, {}
    with open(word_path, encoding='utf-8') as f1, \
            open(pos_path, encoding='utf-8') as f2, \
            open(tag_path, encoding='utf-8') as f3:
        data1 = f1.readlines()   # 读取所有行，返回list
        data2 = f2.readlines()
        data3 = f3.readlines()
        for index, item in enumerate(data1):
            word2id[item.strip()] = index       # 如果不strip()，item会有换行符\n
        for index, item in enumerate(data2):
            pos2id[item.strip()] = index        # 下标从1开始，0作为填充
        for index, item in enumerate(data3):
            tag2id[item.strip()] = index
            id2tag[index] = item.strip()
    return word2id, pos2id, tag2id, id2tag

