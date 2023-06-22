import spacy
import en_core_web_sm
from spacy.symbols import nsubj, VERB, dobj,pobj


import json
from src.utils import from_json,to_json
from os.path import realpath, dirname
from os.path import join as j

nlp = spacy.load("en_core_web_sm", disable=["tok2vec","lemmatizer","attribute_ruler"])
output_path = realpath(j(dirname(__file__), 'training_data'))

def entity(adict):

    text = list(adict['text'])
    _id = list(adict['id'])
    out= {}
    counter = 0

    for doc in nlp.pipe(text, batch_size=20, n_process =-1):
        #for doc in docs:        
        entities = doc.ents
        #print(doc.to_json())
        print(entities.__repr__())
        break

        verbs = set()
        subject = set()
        obj = set()

        for token in doc.to_json():
            if token.dep == nsubj and token.head.pos == VERB:
                verbs.add(token.head)
            if token.dep == nsubj:
                subject.add(token)
            if token.dep == pobj or dobj:
                obj.add(token)
        
        out.update({_id[counter]:{"Entities": list(entities), "verbs":list(verbs),"subject":list(subject),"object":list(obj)}})

        counter += 1
        if counter >2:
            break
    
    return out

    #yield [(token.text, token.tag, token.pos)]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-filepath', help="path to file")
    args = parser.parse_args()
    file = args.filepath

    news= from_json(file)
    sentiment = from_json("training_data/train.json")

    OUTPUT = entity(news)
    print(OUTPUT.to_json())
    #for key,value in sentiment.items():
        #temp = OUTPUT.get(key)
        #temp['sentiment'] = sentiment.get(key)
        #OUTPUT.update(temp)

    to_json(OUTPUT, j(output_path, 'parsed_tagged_train.json'))
   

    

    
