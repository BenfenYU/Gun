from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from LabelAccuracyEvaluator import *
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys, config
from self_dataset import *

def train_self():
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
    #model_name = 'bert-base-multilingual-uncased'
    model_name = './pretrained_model/bert-base-chinese'
    train_batch_size = config.train_batch_size

    self_reader = Self_csv_DataReader('./self_dataset')
    train_num_labels = config.train_num_labels
    model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name,cache_dir = './pretrained_model')

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    # Convert the dataset to a DataLoader ready for training
    logging.info("Read AllNLI train dataset")
    train_dataset = SentencesDataset(examples=self_reader.get_examples("train.csv"), model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)



    logging.info("Read STSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=self_reader.get_examples('dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = LabelAccuracyEvaluator(dev_dataloader,softmax_model = Softmax_label(model = model,
                                                                                    sentence_embedding_dimension = model.get_sentence_embedding_dimension(),
                                                                                    num_labels = train_num_labels))
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    # Configure the training
    num_epochs =1 

    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))



    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=100,
            warmup_steps=warmup_steps,
            output_path=model_save_path
            )

def train_nli():

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    #model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
    model_name = './pretrained_model/bert-base-uncased'

    # Read the dataset
    train_batch_size = 6
    nli_reader = NLIDataReader('./examples/datasets/AllNLI')
    sts_reader = STSBenchmarkDataReader('./examples/datasets/stsbenchmark')
    train_num_labels = nli_reader.get_num_labels()
    model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_name)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    # Convert the dataset to a DataLoader ready for training
    logging.info("Read AllNLI train dataset")
    train_dataset = SentencesDataset(nli_reader.get_examples('train.gz'), model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)



    logging.info("Read STSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = LabelAccuracyEvaluator(dev_dataloader,softmax_model = Softmax_label(model = model,
                                                                                    sentence_embedding_dimension = model.get_sentence_embedding_dimension(),
                                                                                    num_labels = train_num_labels))

    # Configure the training
    num_epochs = 1

    warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))



    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=100,
            warmup_steps=warmup_steps,
            output_path=model_save_path
            )



    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    #model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
    #evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

    model.evaluate(evaluator)

if __name__ == '__main__':
    train_nli()
