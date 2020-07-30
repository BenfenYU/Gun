##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from LabelAccuracyEvaluator import *
from sentence_transformers.readers import *
import logging, sys
from datetime import datetime
from self_dataset import *
import config

def test_self():
    sts_reader = Self_csv_DataReader('./self_dataset/')
    model_save_path = './output'
    dir_list = os.listdir(model_save_path)
    #对文件修改时间进行升序排列
    dir_list.sort(key=lambda fn:os.path.getmtime(model_save_path+'/'+fn))
    #获取文件所在目录
    model_save_path = os.path.join(model_save_path,dir_list[-1])

    model = SentenceTransformer(model_save_path)
    #test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    test_data = SentencesDataset(examples=sts_reader.get_examples("test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.train_batch_size)
    #evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    evaluator = LabelAccuracyEvaluator(test_dataloader,softmax_model = Softmax_label(model = model,
                                                                                    sentence_embedding_dimension = model.get_sentence_embedding_dimension(),
                                                                                    num_labels = config.train_num_labels))

    acc = model.evaluate(evaluator,output_path = model_save_path)
    #print(acc)

if __name__ == '__main__':
    test_self()

