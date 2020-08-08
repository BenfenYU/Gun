from torch.utils.data import DataLoader
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from LabelAccuracyEvaluator import *
from sentence_transformers.readers import *
import logging, sys
from datetime import datetime
from self_dataset import *
import config
from sklearn import metrics

def test_self():
    sts_reader = Self_csv_DataReader('./self_dataset/')
    model_save_path = './output'
    dir_list = os.listdir(model_save_path)
    dir_list.sort(key=lambda fn:os.path.getmtime(model_save_path+'/'+fn))
    model_save_path = os.path.join(model_save_path,dir_list[-1])
    model_save_path = './output/training_nli_.-pretrained_model-bert-base-chinese-2020-07-30_15-59-13'

    model = SentenceTransformer(model_save_path)
    examples, label_text = sts_reader.get_examples("test.csv",_eval= True)
    test_data = SentencesDataset(examples=examples, model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.train_batch_size)
    evaluator = LabelAccuracyEvaluator(test_dataloader,softmax_model = Softmax_label(model = model,
                                                                                    sentence_embedding_dimension = model.get_sentence_embedding_dimension(),
                                                                                    num_labels = config.train_num_labels),
                                                                                    label_text=label_text)

    model.evaluate(evaluator,output_path = model_save_path)
    #print(acc)

def for_learn():
    #self_reader = Self_csv_DataReader('./self_dataset')
    #self_reader.get_examples("train.csv")
    nli_reader = NLIDataReader('examples/datasets/AllNLI')
    nli_reader.get_examples("train.gz")

if __name__ == '__main__':
    test_self()

