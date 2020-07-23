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
        model_path = "birnn_no_pack_1900epoch.pth"
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

if __name__ == '__main__':
    cal = eval_dev()
    cal.decode_to_file()
