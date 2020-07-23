# 存放配置 ，方便后期同意调参
import torch
from build_dict import *
class Config(object):
    # file path
    train_words_path = './data/train/a.txt'     # 训练数据路径
    train_pos_path = './data/train/b.txt'   # 训练数据词性路径
    train_tag_path = './data/train/c.txt'   # 训练数据标签数据s

    # dev用于测试的路径
    dev_words_path = './data/dev/a.txt'     # 验证集待训练数据
    dev_pos_path = './data/dev/b.txt'
    dev_pred_id_path = './data/dev/prediction_id.txt'  # 验证集预测tag：id形式
    dev_cpb = './data/dev/cpbdev.txt'   # 验证集原始数据，用来计算评价指标对比

    path_tag_id = './data/cal_dev/tag_id.txt'
    # 将预测的id形式tag转换成str形式
    path_tag_str = './data/cal_dev/tags_str.txt'
    # 给tag加上BIES前缀
    path_tag_bies = './data/cal_dev/tags_BIES.txt'
    # 最终形态的str形式预测文件
    path_tag_preds = './data/cal_dev/preds.txt'
    # 用于生成最终形式的文件
    dev_file = './data/cal_dev/dev.txt'
    # 正确的标注文件
    gold_file = './data/cal_dev/cpbdev.txt'

    # model保存路径
    # save_path = './data/model/models'     # model保存路径

    # model params
    word_emb_dim = 100
    pos_emb_dim = 10
    hidden_dim =120
    n_voc = 14000  # 词数量
    n_tag = 20  # 标签数 20+2
    n_pos = 33  # 词性数
    lr = 2e-4

    # train params
    epoch = 3001
    batch_size = 1
    num_works = 4 # 线程数

    # dict
    word2id, pos2id, tag2id, id2tag = Build_dict()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_freq = 100
