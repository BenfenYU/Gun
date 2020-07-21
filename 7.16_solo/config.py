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
    epoch = 1000
    batch_size = 256
    num_works = 4 # 线程数

    # dict
    word2id, pos2id, tag2id, id2tag = Build_dict()

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss_lamda = 0.001
    weight = torch.tensor([63.7913,0.00013508923882291155,0.0009514557169704382,0.0017777087281239548, 0.003575945961096474, 0.0057366276978417265, 0.006872581340228399, 0.009030478482446206, 0.009627422275882886, 0.014739209796672828, 0.03066889423076923, 0.051361755233494365, 0.07183704954954954, 0.09436582840236686, 0.12244011516314779, 0.11879199255121044, 0.2773534782608696, 0.27261239316239316, 0.7087922222222223, 6.37913])
    save_freq = 100
