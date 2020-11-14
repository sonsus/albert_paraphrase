from config import *
from utils import *


from transformers import AlbertTokenizer
from collections import defaultdict
from pathlib import Path
import jsonlines as jsl
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from math import floor



class PPDataset(Dataset):
    '''

    **** when doing inference, need: --datamode test ****


    '''
    def __init__(self, expconf, getdev=False):
        self.vocab = open_vocab(expconf.vocabpath)
        self.vocabsize = len(self.vocab.itos)
        self.cls, self.sep, self.pad, self.mask = self.vocab.stoi['CLS'], self.vocab.stoi['SEP'],\
                                                    self.vocab.stoi['PAD'], self.vocab.stoi['MASK']
        self.bos, self.eos = self.vocab.stoi['BOS'], self.vocab.stoi['EOS'] #not in use but albert config requires this to be specified
        self.expconf = expconf
        self.datamode = expconf.datamode # mode = train, dev, test
        if getdev:
            self.datamode = 'dev'
        filename = f'{self.datamode}{self.expconf.kfold_k}.jsonl' if self.datamode!='test' else 'test.jsonl'
        if expconf.debug: # for debugging
            filename = f'debug{self.datamode}.jsonl'
        records = list(jsl.open(  Path(self.expconf.dataroot)/filename) )
        processed = defaultdict(list)

        print(f"processing {filename}")
        for record in tqdm(records):
            inp, l = self.build_albert_inputs_label(record, split=self.datamode)
            processed['inputs'].append(inp)
            processed['label'].append(l)

        self._data = processed['inputs']
        self._label = processed['label']

    def __getitem__(self, index):
        return self._data[index], self._label[index] if self.datamode!='test' else self._data[index]

    def __len__(self):
        return len(self._data)

    def collate(self, inputs_labels): # it is for 1split remember.
        batch = defaultdict(list)
        soplabels = []
        datasetids = []
        tor = torch.cuda if torch.cuda.is_available() else torch

        # batch follows the keys in AlbertForPretraining input
        # https://huggingface.co/transformers/v3.1.0/model_doc/albert.html#albertforpretraining
        if self.datamode == 'test':
            for inp in inputs_labels:
                datasetids.append(inp['id'])
                batch['input_ids'].append(tor.LongTensor(inp['mlminput']))
                batch['labels'].append(tor.LongTensor(inp['mlmtarget'])) # this target for MLM not PP
                batch['token_type_ids'].append(tor.LongTensor(inp['typeid']))

        else:
            for inp, l in inputs_labels: # each list(int)
                datasetids.append(inp['id'])
                batch['input_ids'].append(tor.LongTensor(inp['mlminput']))
                batch['labels'].append(tor.LongTensor(inp['mlmtarget']))
                batch['token_type_ids'].append(tor.LongTensor(inp['typeid']))
                soplabels.append(l)

            soplabels = tor.LongTensor(soplabels)

        batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=self.pad).long()
        batch['labels'] = pad_sequence(batch['labels'], batch_first=True, padding_value=self.pad).long()
        batch['token_type_ids'] = pad_sequence(batch['token_type_ids'], batch_first=True, padding_value=1).long()
        batch['attention_mask'] = (batch['input_ids'] != self.pad).float() # masked == 0

        return  Munch(batch), soplabels, datasetids #labels = [] if self.datamode == 'test'

    def build_albert_inputs_label(self, record, split='train'): # split: train dev test
        '''
        record = {'id':int, 's1':list(int), 's2':list(int), 'l':int}

        inputs = {
            'tokens_w_specials':list(int)
            'typeids':list(int)
            'attention_mask' --> later in collate()
        }

        if split == train --> add augmented to inputs(exchange s1 and s2)
        '''
        tokens_w_specials = [self.cls]
        tokens_w_specials.extend(record['s1'])
        tokens_w_specials.append(self.sep)
        tokens_w_specials.extend(record['s2'])
        tokens_w_specials.append(self.sep)

        typeid = [0 for i in range(len(record['s1']) + 2 )]
        while len(typeid)<len(tokens_w_specials):
            typeid.append(1)

        inputs = dict()
        inputs['id'] = record['id']
        inputs['tokens_w_specials'] = tokens_w_specials

        inputs['typeid'] = typeid
        inputs['mlminput'], inputs['mlmtarget'] = self.masking(tokens_w_specials)

        '''# should do augmentation on jsonl, not here (augmented pair included in same minibatch)
        if split == 'train':
            t_w_s1 = [self.cls]
            t_w_s1.extend(record['s2'])
            t_w_s1.append(self.sep)
            t_w_s1.extend(record['s1'])
            t_w_s1.append(self.sep)
            inputs['t_w_s1'] = t_w_s1
            inputs['mlminput1'], inputs['mlmtarget1'] = self.masking(expconf, t_w_s1)'''


        if split == 'test':
            return inputs

        label = record['l']
        return inputs, label

    def masking(self, inputids):
        '''
        inputids = list[int]
        return: list[int] (0,1)
        '''
        tor = torch.cuda if torch.cuda.is_available() else torch
        inputids = tor.LongTensor(inputids) # shape = (L,)

        ratio = self.expconf.maskratio
        #nummask = floor(ratio * L)

        L = inputids.numel()
        if self.expconf.masking=='span': # heuristic implementation
            # pick starting positions with ratio = maskratio / Mean(number_masked)
            # for each picked starting, sample from Geo(ngram)
            # mask! (~nearly $ratio % masked)
            ps = [1/i for i in range(1, self.expconf.span_n+1) ]
            norm = sum( ps )
            ps_ = [p_/norm for p_ in ps]
            p_ratio = self.expconf.span_n / norm
            mask = tor.BoolTensor(np.random.choice(2, L, p=[1-ratio, ratio])) # instead, can use torch.bernoulli
            todo = tor.LongTensor( np.random.choice(self.expconf.span_n, L, p=ps_) ) + 1
            mask = todo * mask
            for i,e in enumerate(mask):
                #if e==1:
                #     pass # done already
                if e==2:
                    if i+2<=L:
                        mask[i:i+2]=1
                    else:
                        mask[i]=1
                if e==3:
                    if i+3<=L:
                        mask[i:i+3]=1
                    elif i+2<=L:
                        mask[i:i+2]=1
                    else:
                        mask[i]=1
            mask = mask.bool()

        else: #standard MLM masking randomly
            mask = tor.BoolTensor(np.random.choice(2, L, p=[1-ratio, ratio]))

        mlmtarget = inputids.clone().detach().masked_fill_(~mask, -100) #for crossentropy ignorance

        mlminput = inputids.clone().detach()
        mlminput = torch.where(mask, -tor.FloatTensor(np.random.random(L))-1, mlminput.float() ).float()
        mlminput = torch.where( (-1.8<mlminput) & (mlminput<=-1), self.mask * torch.ones_like(mlminput), mlminput) # 80% == MASK

        mlminput = torch.where( (-1.9<mlminput) & (mlminput<=-1.8), inputids.float(), mlminput ) # 10% random, 10% remain the same

        randtokens = torch.randint(self.vocabsize-6, (L,)).float().cuda() if tor == torch.cuda else torch.randint(self.vocabsize-4, (L,)).float()
        mlminput = torch.where( (-2<mlminput) & (mlminput<=-1.9), randtokens, mlminput ) # 10% random, 10% remain the same

        return mlminput.long(), mlmtarget


def get_loader(expconf, getdev=False):
    print("preparing dataloader")

    ds = PPDataset(expconf, getdev=getdev)

    print(f"\tafter calling dataset: {ds.datamode}")
    print(f"\t\tds.datamode: {ds.datamode}")

    print(f"preparing dataloader for {ds.datamode}{expconf.kfold_k}.jsonl")
    loader = DataLoader(dataset = ds,
                        batch_size = expconf.bsz,
                        shuffle = (ds.datamode == 'train'),
                        num_workers = expconf.numworkers,
                        collate_fn = ds.collate,
                        drop_last = True # remaining tensor of size 1(minibatch size == 1) exception ==> drop
                        )


    return list(loader), ds.vocab, ds # ds instance for later specials use
