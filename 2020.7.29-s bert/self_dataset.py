import csv
import gzip
import os
from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))

class Self_DataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)

    Default values expects a tab seperated file with the first & second column the sentence pair and third column the score (0...1). Default config normalizes scores from 0...5 to 0...1
    """
    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=False, min_score=0, max_score=5):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with gzip.open(filepath, 'rt', encoding='utf8') if filename.endswith('.gz') else open(filepath, encoding="utf-8") as fIn:
            data = csv.reader(fIn)#, delimiter=self.delimiter, quoting=self.quoting)
            next(data)
            examples = []
            warm_num = 0
            for id, row in enumerate(data):
                #if warm_num == 100:
                #   break
                #warm_num += 1

                score = int(row[self.score_col_idx]) - 1 
                if self.normalize_scores:  # Normalize to a 0...1 value
                    score = (score - self.min_score) / (self.max_score - self.min_score)

                s1 = row[self.s1_col_idx]
                s2 = row[self.s2_col_idx]
                examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=score))

                if max_examples > 0 and len(examples) >= max_examples:
                    break

        return examples

class Self_csv_DataReader(Self_DataReader):
    """
    Reader especially for the STS benchmark dataset. There, the sentences are in column 5 and 6, the score is in column 4.
    Scores are normalized from 0...5 to 0...1
    """
    def __init__(self, dataset_folder, s1_col_idx=3, s2_col_idx=4, score_col_idx=1, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=False, min_score=0, max_score=5):
        super().__init__(dataset_folder=dataset_folder, s1_col_idx=s1_col_idx, s2_col_idx=s2_col_idx, score_col_idx=score_col_idx, delimiter="\t",
                 quoting=quoting, normalize_scores=normalize_scores, min_score=min_score, max_score=max_score)