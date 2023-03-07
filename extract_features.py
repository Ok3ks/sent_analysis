from src.utils import spacy_preproc, parse_html, extract_features, regex_preproc
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pickle
import json


def extract_and_pickle(args):
    r"""Extract features from text, and pickle vectors"""

    with open(args.path, 'r') as ins:
        obj = json.load(ins)

    filename= args.path.split('/')[-1].split('.')[0]

    processed = [regex_preproc(text['review']) for text in obj]
    labels = [label2id[item['sentiment']] for item in obj] 
    vec,feat,vectorizer = extract_features(processed)

    with open(f'data/pickle/{filename}_vec.pkl', 'wb') as out:
        pickle.dump(vec, out)

    with open(f'data/pickle/{filename}_feat', 'w') as out:
        for i in feat:
            out.write(i+',')

    with open(f'data/pickle/{filename}_vectorizer.pkl', 'wb') as out:
        pickle.dump(vectorizer, out)

    with open(f'data/pickle/{filename}_label.pkl', 'wb') as out:
        pickle.dump(labels, out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help = 'path to text file')
    args = parser.parse_args()
    extract_and_pickle(args)
    