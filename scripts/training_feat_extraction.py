from src.utils import spacy_preproc, parse_html, extract_features, regex_preproc
from sklearn.feature_extraction.text import TfidfVectorizer

from os.path import join
from src.paths import DATA_DIR

import re
import pickle
import json

label2id = {"positive": 1, "negative" : 0}

def extract_and_pickle(file):
    r"""Extract features from text, and pickle vectors"""

    assert os.path.isfile(), 'File does not exist'
    with open(file, 'r') as ins:
        obj = json.load(ins)     
    filename= args.split('/')[-1].split('.')[0]

    processed = [regex_preproc(text['review']) for text in obj]
    vec,feat,vectorizer = extract_features(processed)

    with open(f'{join(DATA_DIR, filename)}_vec', 'wb') as out:
        pickle.dump(vec, out)
    with open(f'{join(DATA_DIR, filename)}_feat', 'w') as out:
        for i in feat:
            out.write(i+',')
    with open(f'{join(DATA_DIR, filename)}_vectorizer.pkl', 'wb') as out:
        pickle.dump(vectorizer, out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to text file')
    args = parser.parse_args()
    extract_and_pickle(args.path)
    #create support for dictionary entry
    