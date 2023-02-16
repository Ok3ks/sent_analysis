from src.utils import PrepareCorpus, AssessData
from src.paths import DATA_DIR, CHUNK_DIR
from os.path import join
import os
import csv

def _chunk(ab):
    file = PrepareCorpus(ab)
    corpus = file._prep()
    for topic in os.listdir(ab):
        temp_dict = {i:j for i,j in enumerate(corpus[topic])}
        output = AssessData(temp_dict, corpus[topic])
        output = output._chunk(200)
        with open(join(CHUNK_DIR, ab.split('-')[-1], f"{topic+ '.' + 'chunk'}.csv"), 'w', newline= '') as inp:
            man = csv.writer(inp)
            man.writerows(output.items())
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type= str, help = 'path to unchunked file')
    args = parser.parse_args()
    output = _chunk(args.filepath)