
import os
import json
import random
import pickle 


from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR, MODEL_DIR
from src.utils import load_config, to_json,regex_preproc
from metaflow import FlowSpec, step, Parameter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import logging

#from metaflow import flowspec

#Modify code into flowspec

def load_system():
    """Loads saved model and vectorizer, returns vectorizer, model"""
    with open('data/pickle/train_vectorizer.pkl', 'rb') as ins:
        train_vectorizer = pickle.load(ins)
    
    with open(join(MODEL_DIR,'svm.pkl'), 'rb') as out:
        model = pickle.load(out)

    return train_vectorizer, model

def test_model(filepath):

    id2label = {1:"positive", 0:"negative"}

    if os.path.isfile(filepath):
        with open (filepath, 'r') as ins:
            text = ins.readlines()
        processed = [regex_preproc("".join(text))] 
    else:
        processed = [regex_preproc(filepath)]
    #print(processed)

    train_vectorizer,model = load_system()
    test_vec = train_vectorizer.transform(processed)

    #access test file
    pred_svm = model.predict(test_vec)
    pred_svm = pred_svm.tolist()
    #print(pred_svm)
    result = {"sentiment": id2label.get(item) for item in pred_svm}
    
    #print(result)
    return result

def save_results():
    """Saves result"""
    log_file = join(LOG_DIR, 'results_svm.json')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    to_json(results, log_file)
    logging.info(f"(Saving model to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type= str, help = 'path to json comment file')
    args = parser.parse_args()
    test_model(args.filepath)

