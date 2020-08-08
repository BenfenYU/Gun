from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.SentenceTransformer import *
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sentence_transformers.util import batch_to_device
import os
import csv
import torch.nn as nn
from sklearn import metrics

class Softmax_label(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        super(Softmax_label, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor,\
        concatenation_sent_rep = True,concatenation_sent_difference = True,concatenation_sent_multiplication = False):
        reps = []
        for sentence_feature in sentence_features:
            feature = self.model(sentence_feature)
            emb = feature['cls_token_embeddings']
            #emb = feature['sentence_embedding']
            reps.append(emb)
        #reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        
        return reps, output


class LabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model = None, label_text = None):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.softmax_model = softmax_model
        self.softmax_model.to(self.device)
        self.label_text = label_text

        if name:
            name = "_"+name

        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.csv_label_text = "label_and_text" + name + "_results.csv"
        self.label_text_headers = ["real","prediction","arg_1","arg_2"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate

        pre_results = torch.tensor([],dtype=torch.int64).to(self.device)

        prf = torch.tensor([],dtype=torch.int64).to(self.device)
        labels = torch.tensor([],dtype=torch.int64).to(self.device)
        for step, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            #pre = torch.argmax(prediction, dim=1)
            #prf = torch.cat((prf,pre))
            #labels = torch.cat((labels,label_ids))
            #correct += pre.eq(label_ids).sum().item()
            
            #if self.label_text:
            #    pre_results = torch.cat((pre_results,pre), 0 )
        
        #prf = prf.view(-1)
        #labels = labels.view(-1)

        #p = metrics.precision_score(labels.cpu(), prf.cpu(), average=None)
        #r = metrics.recall_score(labels.cpu(), prf.cpu(), average=None)
        #f = metrics.f1_score(labels.cpu(), prf.cpu(), average=None)

        #target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
        #prf_result = metrics.classification_report(labels.cpu(), prf.cpu(), target_names=target_names)

        #print(p,r,f)
        #print(prf_result)

        accuracy = correct/total

        logging.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        print("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])
        
        if self.label_text :
            pre_results = pre_results.cpu().numpy().tolist()
            for i in range(len(self.label_text)):
                self.label_text[i].insert(1,pre_results[i])
            
            if output_path is not None:
                csv_path = os.path.join(output_path, self.csv_label_text)
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.label_text_headers)
                    for element in self.label_text:
                        writer.writerow(element)

        return accuracy