from metaflow import FlowSpec,Parameter,step

from src.paths import MODEL_DIR,CONFIG_DIR,DATA_DIR,LOG_DIR
from src.utils import to_json
from src.dataset import IMDB
from src.utils import regex_preproc
from sklearn.feature_extraction import TFidfVectorizer

from pytorch_lightning.loggers import TensorBoardLogger

from sklearn import svm
from os.path import realpath, join

import pickle
import os
import numpy as np 
import random
import json


class EvalClassifier(FlowSpec):

    model_path = Parameter('model', help = "path to pickled model", default = join(MODEL_DIR, 'svm.pkl'))
    #config_path = Parameter('model', help = "path to model config", default = join(CONFIG_DIR, 'svm'))
    vectorizer_path = Parameter('vectorizer', help = "path to vectorizer", default = join(DATA_DIR, 'pickle/train_vectorizer.pkl'))
    test_path = Parameter('input', help = "path to test json", default = join(DATA_DIR, 'test.json') )
    
    @step
    def start(self):
        r"""Set random seeds for reproducibility"""

        random.seed(42)
        np.random.seed(42)
        print("Initialization complete, now loading system")
        self.next(self.load_system)

    @step 
    def load_system(self):
        """Loads system from pickled model"""

        print("Loading system")
        with open(self.model_path, "rb") as ins:
            self.model = pickle.load(ins)
            
        with open(self.vectorizer_path, 'rb') as ins:
            self.vectorizer = pickle.load(ins)

        self.system = Pipeline([('vectorizer', TFidfVectorizer(**vectorizer_params)), ('model', self.model())])
        self.trainer = Trainer(logger = TensorBoardLogger, save_dir = LOG_DIR)
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Evaluates on test set"""
        
        with open(self.test_path, 'r') as ins:
            self.test_json = json.load(ins)

        self.trainer.test(self.system, self.test_json)
        self.preds = self.trainer.test_results
        
        print("Loaded test Json")
        self.test_json = [regex_preproc(text['review']) for text in self.test_json]
        

        log_file = join(LOG_DIR, 'test.json')
        os.makedirs(log_file, exist_ok = True)
        to_json(self.preds, log_file)

        self.next(self.end)

    @step
    def end(self):
        "End of node"
        print("Process complete")

if __name__ == "__main__":

    flow = EvalClassifier()