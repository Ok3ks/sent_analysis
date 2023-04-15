import os
from src.paths import DATA_DIR,CONFIG_DIR
import json
from dotmap import DotMap
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from os.path import join

import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from typing import List


label2id = {"positive": 1, "negative" : 0}

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
    params = vectorizer.get_params()

    print(vec.shape)
    return vec, X_features, vectorizer

class AssessData():
    """Class accepts two dictionaries, one which indexes every string entry, the other,a dictionary of lists
     with labels/classes as keys """
    def __init__(self, dictstringindex: dict, adict:dict):
        
        #Processes list of strings, 
        self._content = dictstringindex
        self._adict = adict

    def _label_to_index(self):
        """Converts labels into index"""
        l_2_idx = {}; index = 0
        for key,alist in self._adict.items():
          for x in alist:
            l_2_idx[index] = key
            index += 1
        return l_2_idx

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
            
    def _create_distribution(self, threshold = {'longformer': 4096, 'BERT': 512, "Short" : 300 }): 
        "Creates a distribution based on the number of words in text strings"
        
        self._get_string_length()
        self._threshold = threshold
        self._distribution = {}
        self._distribution = {key:self._distribution.get(key, []) for key,_ in self.threshold}

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

        if self._distribution is None :
            self._create_distribution()

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

        assert self._pointer < 250 , "Segments should be less than 250"
        assert self._pointer + self._shift < 512, "Should be less than limits of BERT-base"

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
                      content = temp[count*self._pointer - self._shift - 1: count+1*self._pointer]
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

def prep(directory):
    """Prepares a dictionary of lists from a directory path, no preprocessing. Name of directory corresponds to dictionary key
    Single level of directory,files should exist in next sub-directory"""
    
    corpus = {}; temp = []

    assert os.path.isdir(directory) == True, 'Directory does not exist'
    assert os.listdir(directory) == List[str], 'Directory is empty'

    for topic in os.listdir(directory):
        subfolder = str(directory) + '/' + topic
        current = []
        if os.path.isdir(subfolder):
            for doc in os.listdir(subfolder):
                if str(doc) == '.DS_Store':
                    pass 
                else:
                    file = subfolder + '/' + doc
                with open(file, 'r', encoding='utf-8', errors= 'ignore') as t:
                    temp = " ".join(t.readlines())
                    current.append(temp)
            corpus[topic] = current
        else:
            with open(file, 'r', encoding='utf-8', errors= 'ignore') as t:
                    temp = " ".join(t.readlines())
                    current.append(temp)
            corpus[topic] = current    
    return corpus

def to_json(x, filepath):
  with open(filepath, 'w') as fp:
    json.dump(x, fp)

def from_json(filepath):
  with open(filepath, 'r') as fp:
    data = json.load(fp)
  return data

def load_config(config_path):
  return DotMap(from_json(config_path))

