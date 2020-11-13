from fire import Fire
import jsonlines as jsl
from tqdm import tqdm
#import csv
#import json
#from collections import defaultdict
from munch import Munch
from pathlib import Path


def merge(k=0, dataroot='./'):#, vocabpath='./vocab.json'):
    dataroot = Path(dataroot)
    with jsl.open(dataroot/f"train{k}.jsonl") as trainaug, jsl.open(dataroot/f"dev{k}.jsonl") as dev, jsl.open('train.jsonl', 'w') as writer:
        #vocab = json.load(v)
        # make splits
        trainaug = list(trainaug)
        dev = list(dev)

        Lt = len(trainaug)
        Ld = len(dev)
        assert Lt/2 + Ld == 40000, f"wc -l trainaug == 36000*2 if prepdata properly augmented\ntrainaug={Lt}\tdev={Ld}"

        for obj in tqdm(dev):
            obj1 = obj.copy()
            obj, obj1= Munch(obj), Munch(obj1)
            obj1.s1 = obj.s2
            obj1.s2 = obj.s1 # augmentation
            writer.write(obj.toDict())
            writer.write(obj1.toDict())

        for obj in tqdm(trainaug):
            writer.write(obj)


if __name__ == '__main__':
    Fire(merge)
    '''
    python merge_traindev4sub.py
    '''
