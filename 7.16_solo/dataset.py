'''
自定义dataset类和重写collate_fn
'''
from torch.utils.data import Dataset
from config import Config
import torch
from torch.nn.utils.rnn import pad_sequence

config = Config()

# 把words和tags组成一个元组，所有元组组成列表？
def zip_dataset(train = True):
    if not train:
        words_path = config.dev_words_path
        pos_path = config.dev_pos_path
        tags_path = './data/dev/c.txt'
    else:
        words_path = config.train_words_path
        pos_path = config.train_pos_path
        tags_path = config.train_tag_path
    dataset = []
    with open(words_path, encoding='utf-8') as f1, \
            open(tags_path, encoding='utf-8') as f2, \
            open(pos_path, encoding='utf-8') as f3:
        words_data, tags_data, pos_data = f1.readlines(), f2.readlines(), f3.readlines()
        words, tags, pos =[], [], []
        for line in words_data:  # 消除每行的换行符
            line = line.strip()
            words.append(line)
        for line in tags_data:
            line = line.strip()
            tags.append(line)
        for line in pos_data:
            line = line.strip()
            pos.append(line)
        dataset = list(zip(words, pos, tags))
    return dataset

'''一行words或者对应tags转换成对应id，并且返回list'''
def to_id_list(words, w2id):
    # words是list格式
    return [w2id[w] if w in w2id else w2id['_PAD'] for w in words]

def to_id_list2(words, w2id):
    # words是list格式
    tag = []
    for o in words:
        w = torch.zeros(config.n_tag).numpy().tolist()
        w[w2id[o]] = 1
        tag.append(w)

    return tag

# 自定义数据集，需要重写Dataset的两个方法
# 使用方法：把数据路径传入ModelDataset，通过ModelDataset处理数据
# dataset的形状：words和tags组成
# model的训练，forward方法输入的是tensor形式的batch，Dataset返回的应该是tensor形式
'''
# list_1 = ['宋浩京 转达 了 朝鲜 领导人 对 中国 领导人 的 亲切 问候 ， 代表 朝方 对 中国 党政 领导人 和 人民 哀悼 金日成 主席 逝世 表示 深切 谢意 。', '中国 爱 人民']
# list_2 = ['O O O O O O O O O O O O O O O ARG0 ARG0 ARG0 ARG0 ARG0 rel ARG1 ARG1 ARG1 O O O O', 'o o o']
# dataset = list(zip(list_1, list_2))
# for x in dataset:
#     print(x[0])
#     print(x[1])
'''
'''因为返回的数据要组成batch，然后把batch输入已经训练好的词向量二维表，返回batch里面单词对应的词向量。注意，pytorch的Embedding只能输入word对应的index'''
class MyDataset(Dataset):
    def __init__(self, dataset,to_id = True):
        '''
        dataset：函数zip_dataset返回的dataset
        dataset的形状：一个list，每个元素是一个元组，一行words 以及对应的tags
        '''
        self.dataset = dataset
        self.words, self.pos, self.tags, self.lengths = [], [], [],[]
        for line in dataset:
            ''' 把字符串line转换成list，然后用to_list()转换成tensor形式的对用id，当循环完成每个words'''
            self.words.append(to_id_list(line[0].split(' '), config.word2id))
            self.pos.append(to_id_list(line[1].split(' '), config.pos2id))
            self.tags.append(to_id_list(line[2].split(' '), config.tag2id))
            self.lengths.append(len(self.words[-1]))
            # 把dataset转换成list形式的对应id序列

    # 返回数据集的一个样本：一行words以及对应的pos, tags。tensor形式
    def __getitem__(self, item):
        return torch.tensor(self.words[item]),  torch.tensor(self.pos[item]), torch.tensor(self.tags[item]) ,torch.tensor(self.lengths[item])

    # 返回数据集的大小：有几行words
    def __len__(self):
        return len(self.words)

# 重写Dataloader的collate_fn方法
# 对传入的一个batch进行处理：例如padding,把batch中每一句话填充至最大长度（当前batch的最大长度）
# 遍历list的每一个元素：元组，其中下标0表示元组第一个元素，下标1表示元组第二个元素
def collate_fn(batch):
    '''batch：一个batch有batch_size个words，每个words长度不等，补齐和最长words一致。'''
    '''__getitem__返回两个张量，分别是每行words对应id，以及tags对应id，dict从1开始，长度不足的填充0'''
    '''line[0]是tensor，最终words是batch里面words列表，每个元素是tensor形式。 例如：[(tensor([ 288, 2    3, 8945,    5]), tensor([2, 2, 2, 2, 2, 2]))]'''
    words_list = [line[0] for line in batch]  # 以tensor为元素的列表
    pos_list = [line[1] for line in batch]
    tags_list = [line[2] for line in batch]
    # 每行words的长度
    len_list = [len(line[0]) for line in batch]
    # pad_sequence的传参是list，元素是tensor形式。不能list形式，会报错。所以Mydataset的__getitem__返回tensor
    words_list = pad_sequence(words_list, padding_value=0, batch_first=True)
    pos_list = pad_sequence(pos_list, padding_value=0, batch_first=True)
    tags_list = pad_sequence(tags_list, padding_value=0, batch_first=True)
    # 需要倒置 transposr(0, 1)
    # print(words_list)
    return words_list.to(config.device), pos_list.to(config.device), tags_list.to(config.device), len_list  # 最终返回tensor，每个元素是一行填充过的words的id


