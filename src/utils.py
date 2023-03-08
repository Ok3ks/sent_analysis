import os
from src.paths import DATA_DIR
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split

import transformers
from transformers import get_linear_schedule_with_warmup
import time
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

def compute_metrics(predictions,label):

    acc = load_metric('accuracy')
    f1_score = load_metric('f1')

    f_1 = f1_score.compute(predictions = predictions, references = label, pos_label = 1)
    acc_1 = acc.compute(predictions = predictions, references = label)
    return acc_1['accuracy'], f_1['f1']

def parse_html(text):
    "removes hyperlink from a piece of text"
    
    soup = BeautifulSoup(text, "html.parser")
    parsed_text = soup.get_text()
    return parsed_text


def spacy_preproc(text):

    "Removes stopwords and punctuation and"
    nlp = English()
    tokenizer = nlp.tokenizer
    refined_text = tokenizer(text)
        
    refined_text = [text for text in refined_text if not text.is_stop]
    refined_text = [text for text in refined_text if not text.is_punct]
    refined_text = [text for text in refined_text if not text.like_email]
    refined_text = [text for text in refined_text if not text.like_url]
    refined_text = [text.text for text in refined_text if not text.like_num]

    text = " ".join(refined_text[:])

    return text

def regex_preproc(text): 

    refined_text = re.sub("(\n)", " ", text) #removes python newline 
    refined_text = re.sub("\w+(\')", "" , refined_text) #removes backslash
    refined_text = re.sub("(<\W+(p >))", " ", refined_text)
    refined_text = re.sub("(\s+)", " ", refined_text) #removes whitespaces
    #refined_text = re.sub("(&amp|>+|-+)", "", refined_text)
    #refined_text = re.sub("\W*(@)", "", refined_text) #removes emails 
    #refined_text = re.sub("(\d+)", "", refined_text) #removes digits

    return refined_text

def extract_features(text, min_df = 0.05 , max_df = 0.5, max_features = 1000, method = "Tfidf"):
    """Count represents the number of features to be chosen from tfidf while text represents text data
    vectorizer is either count_vectorizer or tfidfvectorizer"""
    
    if method.lower() == "tfidf":
        vectorizer = TfidfVectorizer(token_pattern = '[A-Za-z]+', min_df = float(min_df), max_df = float
                                        (max_df), ngram_range = (1,1), max_features = max_features)
    else:
        print("Wrong method. Only Tfidf vectorizer available")

    vec = vectorizer.fit_transform(text)
    X_features = vectorizer.get_feature_names_out()
    print(vec.shape)
    return vec, X_features, vectorizer


class RoBERT_Model(nn.Module):
    """ Make an LSTM model over a fine tuned bert model. Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance
    """

    #added output_class to function

    def __init__(self, config_path, num_classes = 2):
        super(RoBERT_Model, self).__init__()
        config = BertConfig.from_json_file(config_path)
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", config = config)
        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=False)
        self.classifier = torch.nn.Linear(100, num_classes)
        self.out = nn.Softmax(dim= num_classes)
    

    def forward(self, test_set, comb_len):
        """ Define how to performed each call
        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -
        lengt: int

        comb_len: is the dictionary with a key of doc_id and value of the length of each document chunks included
            -
        Returns:
        _______
        -
        """

        #doc_id = chunk["doc_id"]
        output = []

        #doc_tensor = {}

        #doc_ids = [torch.LongTensor(x) for x in test_set["doc_id"][:8]]

        for doc_id,num in comb_len.items():
            start = 0
            ids = [torch.LongTensor(x) for x in test_set["input_ids"][start:start+num]]
            ids = nn.utils.rnn.pad_sequence(ids, batch_first= False)

            mask = [torch.LongTensor(x) for x in test_set["attention_mask"][start:start+num]]
            mask = nn.utils.rnn.pad_sequence(mask, batch_first= False)

            token_type_ids = [torch.LongTensor(x) for x in test_set["token_type_ids"][start:start+num]]
            token_type_ids = nn.utils.rnn.pad_sequence(token_type_ids, batch_first= False)

            out = self.bert.forward(ids, mask, token_type_ids, output_hidden_states=True)
            seq_lengths = torch.LongTensor([x for x in map(len, test_set["input_ids"][start:start+num])])
          
            device = torch.device("cpu")
            out = out.hidden_states[-1].to(device)
            out = out.transpose(0, 1) 

            #lstm_input = nn.utils.rnn.pack_padded_sequence(out, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

            packed_output, (h_t, h_c) = self.lstm(out,)  # (h_t, h_c))
#           output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-91)
            h_t = self.classifier(h_t)
            h_t = self.out(h_t)
            output.append(h_t)
            start = start + num

        return output

class AssessData():
    def __init__(self, dictstringindex: dict, adict:dict):
        
        #Processes list of strings, 
        self._content = dictstringindex
        self._threshold = {'longformer': 4096, 'BERT': 512, "Short" : 300 }
        self._adict = adict

    def _index_to_label(self):
        index_to_label = {}; index = 0
        for key,alist in self._adict.items():
          for x in alist:
            index_to_label[index] = key
            index += 1

        return index_to_label

    def _string_length(self, text):
        """Processes the number of words in one text string """
        if isinstance(text, list):
            print("This is a list, input string")
        else:
            return len(word_tokenize(text))

    def _get_string_length(self):
        """Processes the number of words in a list of text strings"""
        self._all_length = {i: self._string_length(item) for i,item in self._content.items()}
        return self._all_length
            
    def _create_distribution(self): 
        "Creates a distribution based on the number of words in text strings"
        
        self._get_string_length()
        self._distribution = {"Too Long" : [],"Long": [], "BERT": [], "Short":[]}


        for index,length in self._all_length.items():
            if length > self._threshold['longformer']:
                self._distribution["Too Long"].append(index)
            elif length >= self._threshold['BERT'] and length < self._threshold['longformer']:
                self._distribution["Long"].append(index)
            elif length < self._threshold['BERT'] and length >= self._threshold['Short']:
                self._distribution["BERT"].append(index)
            else:
                self._distribution["Short"].append(index)
        
        return self._distribution


    def _extract(self, key = "Longformers"):
        """Extracts text strings suitable for particular classification technique"""

        if key.lower() in self._distribution.keys():
            indexes = self._distribution[key]
            return [self._content[i] for i in indexes]
        else:
            if self._distribution is None: 
                self._create_distribution()
                self._extract(key = key)
            else:
                print("Select from given distribution {}".format(self._distribution.keys()))

    def _visualise(self):
        """Displays a plot of count of number of words versus the categories created"""
        
        values = {k:len(v) for k, v in self._distribution.items()}
        print(values)
        plt.bar(values.keys(),values.values())
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Data Distribution across thresholds")

        plt.show()

    def _chunk(self, words_per_segment, overlap = {"side": "both", "number": 50}):
        
        """Chunking for input into BERT, length of input 
        must be less than 512 with overlap of 50 tokens"""

        self._pointer = words_per_segment
        self._chunks = {}
        self._shift = overlap.get("number")
        self._overlap = overlap

        for index,temp in self._content.items():
            length = len(temp)
            self._chunks[index] = self._chunks.get(index, [])

            if length > self._pointer:
                content = temp[:self._pointer-1]
                count = 1
                self._chunks[index].append(content)

                while count*self._pointer < length:

                    if self._overlap.get('side') == "both":

                      content = temp[count*self._pointer - self._shift//2 -1: (count+1)*self._pointer + (self._shift - self._shift//2)]
                      self._chunks[index].append(content)
                      count +=1                    

                    elif self._overlap.get('side') == "one":
                      content = temp[count*self._pointer - self._shift -1: count+1*self._pointer]
                      self._chunks[index].append(content)
                      count +=1
                    else:
                      print("Choose one or both")


                content = temp[count*self._pointer:]
                self._chunks[index].append(content)

            else:
                content = [temp[:]]
                self._chunks[index].append(content)
        return self._chunks
        


class PrepareCorpus():
  """Prepares a corpus from a list of folders in the local directory
  corresponding to class labels"""

  def __init__(self, path):
        self._path = path
        
  def _prep(self):
        """BY date"""
        self._corpus = {}; temp = []
        for topic in os.listdir(self._path):
            subfolder = self._path + '/' + topic
            current = []
            for doc in os.listdir(subfolder):
                if str(doc) == '.DS_Store':
                    pass 
                else:
                    file = subfolder + '/' + doc
                with open(file, 'r', encoding='utf-8', errors= 'ignore') as t:
                    temp = " ".join(t.readlines())
                    current.append(temp)
            self._corpus[topic] = current
        return self._corpus


def to_json(x, filepath):
  with open(filepath, 'w') as fp:
    json.dump(x, fp)


def from_json(filepath):
  with open(filepath, 'r') as fp:
    data = json.load(fp)
  return data


def load_config(config_path):
  return DotMap(from_json(config_path))

