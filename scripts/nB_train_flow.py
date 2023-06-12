import os
import pytorch_lightning as pl
from os.path import join

import json
import random
import torch
import numpy as np
import pickle

from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR
from src.utils import load_config, to_json,regex_preproc
from metaflow import FlowSpec, step, Parameter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
#from sklearn.feature_extraction import DictVectorizer

#init
#label2id = {"positive": 1, "negative" : 0}

#with open('data/pickle/train_label.pkl', 'rb') as ins:
    #train_label = pickle.load(ins)

#with open('data/pickle/train_vec.pkl', 'rb') as ins:
    #train_vec = pickle.load(ins)

#with open('data/pickle/train_vectorizer.pkl', 'rb') as ins:
    #train_vectorizer = pickle.load(ins)

#MNB = MultinomialNB()
#BNB = BernoulliNB()
#GNB = GaussianNB()

#train_model
#Fitting features and output labels
#MNB.fit(train_vec, train_label)
#BNB.fit(train_vec, train_label)
#GNB.fit(train_vec.toarray(), train_label)

#save model

#with open(join(MODEL_DIR,'mnb.pkl'), 'wb') as out:
    #pickle.dump(MNB, out)

#with open(join(MODEL_DIR,'mnb.pkl'), 'wb') as out:
    #pickle.dump(BNB, out)

#with open(join(MODEL_DIR,'mnb.pkl'), 'wb') as out:
    #pickle.dump(GNB, out)

#access test file
#with open("/workspace/sent_analysis/data/test.json", 'r') as ins:
    #obj = json.load(ins)

#processed = [regex_preproc(text['review']) for text in obj]
#test_label = [label2id[item['sentiment']] for item in obj]

#test_vec = train_vectorizer.transform(processed)

#results = {}
#pred_bnb = BNB.predict(test_vec)
#pred_mnb = MNB.predict(test_vec)
#pred_gnb = GNB.predict(test_vec.toarray())

#results['mnb'] = pred_mnb.tolist()
#results['bnb'] = pred_bnb.tolist()
#results['gnb'] = pred_gnb.tolist()

#present result

#save
log_file = join(DATA_DIR, 'results.json')
os.makedirs(os.path.dirname(log_file), exist_ok = True)
to_json(results, log_file)

#output

