import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import Config

config = Config()

# 对验证集的数据进行处理
def prepare_sequence(seq, to_id):
    idxs = [to_id[w] if w in to_id else to_id['_PAD'] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class TeDataset(Dataset):
    def __init__(self):
        '''把传入的文本，转换成id的list，每个元素是一句话'''
        self.word_path = config.dev_words_path
        self.pos_path = config.dev_pos_path
        with open(self.word_path, 'r', encoding='utf-8') as f1, open(self.pos_path, 'r', encoding='utf-8') as f2:
            self.word_data = [prepare_sequence(line.strip().split(' '), config.word2id) for line in f1.readlines()]
            self.pos_data = [prepare_sequence(line.strip().split(' '), config.pos2id) for line in f2.readlines()]


    def __getitem__(self, item):
        '''返回一条数据，格式：id形式的tensor'''
        return self.word_data[item], self.pos_data[item]

    def __len__(self):
        '''返回本条数据长度'''
        return len(self.word_data)

# 重写Dataloader的collate_fn方法。对传入的tensor形式的一个batch进行处理
def collate_fn_test(batch):
    words = [line[0] for line in batch]
    pos = [line[1] for line in batch]
    length = [len(item) for item in words]

    # 使用pad_sequence进行填充
    pad_word = pad_sequence(words, padding_value=0, batch_first=True)
    pad_pos = pad_sequence(pos, padding_value=0, batch_first=True)
    return pad_word.to(config.device).to(config.device), pad_pos.to(config.device), length

if __name__ == '__main__':
    data = TeDataset()
    print(data[2][0])