import os
from os.path import join

import json
import random
import torch
import numpy as np
import pickle

from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR
from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR, MODEL_DIR
from src.utils import load_config, to_json,regex_preproc, from_json, to_pickle, from_pickle, compute_metrics
from metaflow import FlowSpec, step, Parameter

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
import logging
import pickle

from metaflow import FlowSpec, step, Parameter

label2num = {"positive": 1, "negative" : 0}

class NbTrain(FlowSpec):

    train_path = Parameter('trainfile', 
    help = "Path to train file", 
    default= join(DATA_DIR, 'train.json'))

    variant = Parameter('nbvariant',
    help = "type of naive bayes",default = "mnb")

    @step
    def start(self):
        """Sets path to training data, and TfidfVectorizer"""

        self.train = from_json(self.train_path)
        self.vectorizer = from_pickle(join(DATA_DIR, 'pickle', 'train_vectorizer.pkl'))
        self.model_name = self.variant
        self.next(self.train_model)

    @step
    def train_model(self):
        """Trains SVM model with training data"""

        if self.model_name == "mnb":
            self.model = MultinomialNB()
        elif self.model_name == "gnb":
            self.model = GaussianNB()
        elif self.model_name == "bnb":
            self.model = BernoulliNB()
        else:
            print("""variant has to be either multinomial naive bayes ('mnb'), gaussian naive bayes ('gnb)
            or bernoulli naive bayes ('bnb')""")

        processed = [regex_preproc(text['review']) for text in self.train]
        self.train_label = [label2num[item['sentiment']] for item in self.train]

        self.train_vec = self.vectorizer.transform(processed)
        if self.model_name == "gnb":
            self.model.fit(self.train_vec.toarray(), self.train_label)
        else:
            self.model.fit(self.train_vec, self.train_label)

        self.next(self.end)
    
    @step
    def end(self):

        """Computes training accuracy and saves model for use later on test data"""
        train_pred = self.model.predict(self.train_vec.toarray())
        logging.info(f"Training accuracy is {compute_metrics(train_pred.tolist(), self.train_label)}")

        try :
            os.makedirs(MODEL_DIR, exist_ok=True)  
        except OSError:
            os.makedirs(MODEL_DIR)

        logging.info(f"Saving model to {MODEL_DIR}")
        to_pickle(self.model, join(MODEL_DIR,f"{self.model_name}.pkl"))
        logging.info("Done")

if __name__ == "__main__":
    NbTrain()