
import os
from os.path import join
import json
import random
import pickle 

from src.paths import LOG_DIR, CONFIG_DIR,DATA_DIR, MODEL_DIR
from src.utils import from_json,to_json,regex_preproc,extract_features,compute_metrics
from metaflow import FlowSpec, step, Parameter

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import logging

from metaflow import FlowSpec, step, Parameter

class SvmTest(FlowSpec):

    filepath = Parameter('file', 
                help="path to test file", 
                default = join(DATA_DIR, 'test.json'))

    @step
    def start(self):
        """Loads saved model and vectorizer, returns vectorizer, model"""
        
        if os.path.isfile(self.filepath):
            obj = from_json(self.filepath) 

        with open(join(DATA_DIR, 'pickle', 'train_vectorizer.pkl'), 'rb') as ins:
            train_vectorizer = pickle.load(ins)  

        #self.models = ['svm','mnb','bnb' ,'gnb']
        # setup to test all models at once
          
        with open(join(MODEL_DIR,'svm.pkl'), 'rb') as out:
            model = pickle.load(out)

        self.file = obj
        self.vectorizer = train_vectorizer
        self.model = model
        self.next(self.test_model)

    @step
    def test_model(self):
        """Tests a svm model on comments stored in json"""
        
        id2label = {1:"positive", 0:"negative"} 
        label2id = {"positive":1, "negative":0}

        processed = [regex_preproc(text['review']) for text in self.file]
        self.id_labels =[label2id.get(text["sentiment"]) for text in self.file]

        test_vec = self.vectorizer.transform(processed)
        pred = self.model.predict(test_vec)
        self.pred = pred.tolist()

        sentiment = [{f"{count}": id2label.get(item)} for count,item in enumerate(self.pred)]
        self.result = {"sentiment":sentiment}
        self.next(self.metrics)

    @step
    def metrics(self):
        """Computes accuracy"""
        acc,f_1 = compute_metrics(self.pred,self.id_labels)
        print("Accuracy: {}, f1:{}".format(acc,f_1))
        self.result["accuracy"] = acc
        self.result["f_1"] = f_1
        self.next(self.end)

    @step
    def end(self):
        """Saves result"""
        log_file = join(LOG_DIR, 'result_svm.json')

        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except OSError:
            os.makedirs(os.path.dirname(log_file))

        to_json(self.result, log_file)
        logging.info(f"(Saving model to {log_file}")


if __name__ == "__main__":
    SvmTest()

