
import os
import pytorch_lightning as pl
from os.path import join

import json
import random
import torch
import numpy as np
import pickle 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR, MODEL_DIR
from src.utils import load_config, to_json,regex_preproc
from metaflow import FlowSpec, step, Parameter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import logging

from metaflow import flowspec

#init, use metaflow to systemize
label2id = {"positive": 1, "negative" : 0}

with open('data/pickle/train_label.pkl', 'rb') as ins:
    train_label = pickle.load(ins)

with open('data/pickle/train_vec.pkl', 'rb') as ins:
    train_vec = pickle.load(ins)

with open('data/pickle/train_vectorizer.pkl', 'rb') as ins:
    train_vectorizer = pickle.load(ins)

#Fitting features and output labels
with open("/workspace/sent_analysis/data/test.json", 'r') as ins:
    obj = json.load(ins)

processed = [regex_preproc(text['review']) for text in obj]
test_label = [label2id[item['sentiment']] for item in obj]
test_vec = train_vectorizer.transform(processed)
svc_classifier = svm.LinearSVC()
svc_classifier.fit(train_vec, train_label)

#gridsearch and random search


#save model
logging.info(f"(Saving model to {MODEL_DIR}")

with open(join(MODEL_DIR,'svm.pkl'), 'wb') as out:
    pickle.dump(svc_classifier, out)

#access test file
pred_svm = svc_classifier.predict(test_vec)

results = {}
results['svm'] = pred_svm.tolist()

#save
log_file = join(DATA_DIR, 'results_svm.json')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
to_json(results, log_file)
logging.info(f"(Saving model to {log_file}")

#output

