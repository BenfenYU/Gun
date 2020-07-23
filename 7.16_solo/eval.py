from torch.utils.data import DataLoader
from dev_dataset import *
from dataset import *
import numpy as np
from sklearn.metrics import f1_score,precision_score,accuracy_score
import matplotlib.pyplot as plt
import os

# 本类完成
class eval_dev(object):
    def __init__(self):
        self.word_path = config.dev_words_path  # 待预测数据
        self.pos_path = config.dev_pos_path    # 待预测数据的词性
        self.pred_path = config.dev_pred_id_path  # 预测结果存放路径：id形式的tag

    def decode_to_file(self):
        dataset = TeDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=50,
                                drop_last=False, collate_fn=collate_fn_test)

        # 模型加载
        model_path = "birnn_packed_1400epoch.pth"
        model_name = os.path.splitext(model_path)[0]
        model = torch.load( model_path,map_location=config.device)

        best_path_list = []
        pre_tag = []

        with open(self.pred_path, 'w', encoding='utf-8') as f:
            for batch, pos, len_list in dataloader:
                best_path = model.decode(batch, pos, len_list)      # tensor [batch_size, seq_len]
                tags = [tags[:len_list[index]] for index, tags in enumerate(best_path)]
                pre_tag.extend(to1D(tags))
                
                for item in tags:
                    for i in item:
                        f.write(str(i) + ' ')
                    f.write('\n')
                  
        
        
        test_data = zip_dataset(False)
        tags = MyDataset(test_data).tags
        real_tags = to1D(tags)
        num_d = 0
        x = []
        for i in range(len(real_tags)):
            x.append(i)
                
        print(f1_score(real_tags,pre_tag,average='micro'))
        print(accuracy_score(real_tags, pre_tag))

        plt.figure()
        plt.xlabel('every word has a position')
        plt.ylabel('id of tag')
        plt.scatter(x,real_tags,c = 'red',s = 3, marker='x',label = 'real semantic role')
        plt.scatter(x,pre_tag,c = 'blue', s = 3,marker='o',label = 'prediction of model')
        plt.legend()
        plt.title(model_name)
        plt.savefig(model_name)
        plt.show()

        return

def to1D(newlist):
	d = []
	for element in newlist:
		if not isinstance(element,list):
			d.append(element)
		else:
			d.extend(to1D(element))
	
	return d

import time
from collections import Counter
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists

class Metrics(object):
    """用于评价模型，计算每个标签的精确率，召回率，F1分数"""

    def __init__(self, golden_tags, predict_tags, remove_O=False):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()

        # 辅助计算的变量
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)

        # 计算精确率
        self.precision_scores = self.cal_precision()

        # 计算召回率
        self.recall_scores = self.cal_recall()

        # 计算F1分数
        self.f1_scores = self.cal_f1()

    def cal_precision(self):

        precision_scores = {}
        for tag in self.tagset:
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):

        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2*p*r / (p+r+1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def report_scores(self):
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
        # 打印每个标签的 精确率、召回率、f1分数
        for tag in self.tagset:
            print(row_format.format(
                tag,
                self.precision_scores[tag],
                self.recall_scores[tag],
                self.f1_scores[tag],
                self.golden_tags_counter[tag]
            ))

        # 计算并打印平均值
        avg_metrics = self._cal_weighted_average()
        print(row_format.format(
            'avg/total',
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['f1_score'],
            len(self.golden_tags)
        ))

    def count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)

        # 计算weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size

        for metric in weighted_average.keys():
            weighted_average[metric] /= total

        return weighted_average

    def _remove_Otags(self):

        length = len(self.golden_tags)
        O_tag_indices = [i for i in range(length)
                         if self.golden_tags[i] == 'O']

        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags)
                            if i not in O_tag_indices]

        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags)
                             if i not in O_tag_indices]
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(O_tag_indices),
            len(O_tag_indices) / length * 100
        ))

    def report_confusion_matrix(self):
        """计算混淆矩阵"""

        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        # 初始化混淆矩阵 matrix[i][j]表示第i个tag被模型预测成第j个tag的次数
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)

        # 遍历tags列表
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # 有极少数标记没有出现在golden_tags，但出现在predict_tags，跳过这些标记
                continue

        # 输出矩阵
        row_format_ = '{:>7} ' * (tags_size+1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))

def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./not_ignored_model/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 四个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()



if __name__ == '__main__':
    cal = eval_dev()
    cal.decode_to_file()
