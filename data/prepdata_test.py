from fire import Fire
import csv
import jsonlines as jsl
import json
from collections import defaultdict
from munch import Munch


def open_vocab(vocabpath):
    vocab= Munch(json.load(open(vocabpath)))
    return vocab

def tokenize(vocab, inputs):
    return [vocab.stoi[t] for t in inputs]

def preptest(datapath='./test.csv', vocabpath='./vocab.json'):
    processed = defaultdict(list)
    vocab = open_vocab(vocabpath)

    with open(datapath) as f, jsl.open('test.jsonl', 'w') as tjsl:

        # open csv, jsonl files for splits
        rawt= csv.reader(f)
        processedt = [ (int(id), s1.split(), s2.split()) for (id,s1,s2) in rawt if id != 'id' ]

        for (id,s1,s2) in processedt:
            record = {'id': id, 's1': tokenize(vocab, s1), 's2': tokenize(vocab, s2)}
            tjsl.write(record)


if __name__ == '__main__':
    Fire(preptest)
    '''
    python prepdata_test.py

    if submission=True: all the train.csv --> train.jsonl
    '''
