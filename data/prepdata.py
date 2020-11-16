from fire import Fire
import csv
import jsonlines as jsl
import json
from collections import defaultdict
from munch import Munch

def tokenize(vocab, inputs):
    return [vocab.stoi[t] for t in inputs]

def split_kfold(factor=10, k=3, datapath='./train.csv', vocabpath='./vocab.json'):
    processed = defaultdict(list)
    vocab = list()

    with open(datapath) as f, open(vocabpath, 'w') as v, open('test.csv') as t:

        # open csv, jsonl files for splits
        jsls = [ {'train': jsl.open(f"train{i}.jsonl", 'w'), 'dev': jsl.open(f"dev{i}.jsonl", 'w')} for i in range(k)]
        raw = csv.reader(f) # default option will work properly
        rawt= csv.reader(t)
        processed = [ (int(id), s1.split(), s2.split(), int(label)) for (id,s1,s2,label) in raw if id != 'id' ]
        processedt = [ (int(id), s1.split(), s2.split()) for (id,s1,s2) in rawt if id != 'id' ]

        # make vocab
        for (id,s1,s2,l) in processed:
            vocab.extend(s1)
            vocab.extend(s2)
        for (id, s1, s2) in processedt:
            vocab.extend(s1)
            vocab.extend(s2)

        vocab = list(set(vocab))
        for special in ['CLS', 'SEP', 'PAD', 'MASK', 'BOS', 'EOS']:
            vocab.append(special)

        vl = len(vocab)
        vocab = {
            'stoi': dict( zip(vocab, range(vl)) ),
            'itos': vocab
        }

        json.dump(vocab, v)


        vocab = Munch(vocab)
        print(f"CLS, SEP, PAD, MASK = \
                    {vocab.stoi['CLS'], vocab.stoi['SEP'], vocab.stoi['PAD'], vocab.stoi['MASK']}")

        # make splits
        L = len(processed)
        devidxs = [ list( range( (L//k)*i, (L//k)*i + L//factor ) ) for i in range(k)]

        for (id,s1,s2,l) in processed:
            record = {'id': id, 's1': tokenize(vocab, s1), 's2': tokenize(vocab, s2), 'l': l}
            for i, di in enumerate(devidxs):
                if id-1 in di:
                    jsls[i]['dev'].write(record)
                else:
                    jsls[i]['train'].write(record)
                    record_aug = {'id': id, 's1': tokenize(vocab,s2), 's2': tokenize(vocab, s1), 'l': l}
                    jsls[i]['train'].write(record_aug)

        #close jsonls
        for i in range(k):
            jsls[i]['train'].close()
            jsls[i]['dev'].close()


if __name__ == '__main__':
    Fire(split_kfold)
    '''
    python makesplits.py --factor 10 --k 3 --datapath 'train.csv --submission False'

    if submission=True: all the train.csv --> train.jsonl
    '''
