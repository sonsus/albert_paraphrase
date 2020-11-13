from config import *
from dataload import *
from utils import *
#from schedulers import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW # or import it from
from transformers import AlbertConfig, AlbertForPreTraining, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from munch import Munch
import wandb
import os
import random
import numpy as np
from fire import Fire

def modelconf2transformers(conf):
    return AlbertConfig(
                    num_labels = 2, # for SOP classifier. not sure this is default 2
                    num_hidden_layers = conf['layers'],
                    num_hidden_groups = 1, # shared
                    hidden_size = conf['hidden'],
                    embedding_size = conf['embedding'],
                    max_position_embeddings = conf['max_position_embeddings']
                    )


def evaldev(expconf, model, devloader, ep):
    model.eval()
    L = len(devloader)
    bsz= len(devloader[0])



    lossmlm = 0
    losspp = 0


    for i, (b, l, datasetids) in enumerate(tqdm(devloader, desc="eval iter progress")):
        outputs = model(**b, sentence_order_label=l, return_dict=True)
        vsz= outputs.prediction_logits.shape[-1]
        lossmlm += F.cross_entropy(outputs.prediction_logits.detach().view(-1,vsz).contiguous(), b['labels'].view(-1))
        losspp += F.cross_entropy(outputs.sop_logits, l)

    lossmlm /= L
    losspp /= L

    wandb.log(
    {
            'step': (i + ep*L)*bsz if expconf.see_bsz_effect else (i+ep*L),
            'dev/mlm_loss': lossmlm,
            'dev/pp_loss': losspp,
    } )
    return lossmlm, losspp


def savemodel(expconf, model, vocab, ep, mlm=0, pp=0):
    d_expconf = expconf.toDict()
    saveroot = Path(expconf.modelsaveroot)
    todaydir = saveroot / get_date()
    if not todaydir.is_dir():
        Path.mkdir(todaydir, parents=True)

    savename = f"albert.pp{pp}.mlm{mlm}.m{expconf.masking}_{get_time()}_ep{ep}.lr{expconf.lr}.w{expconf.warmups}.sch{expconf.scheduler}.bsz{expconf.bsz}.pth"
    saved = dict()
    saved = {
        'expconf': d_expconf,
        'model': model.state_dict(),
        'vocab': vocab
    }
    savepath = todaydir/savename
    print(f"saving {savename}\n\tat {str(todaydir)}")
    torch.save(saved, savepath)


def main():
    # my dice shows 777 only. period.
    random.seed(EXPCONF.seed)
    np.random.seed(EXPCONF.seed)
    torch.manual_seed(EXPCONF.seed)
    torch.cuda.manual_seed_all(EXPCONF.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader, vocab, _trainds = get_loader(EXPCONF, getdev=False)
    devloader, _, _devds = get_loader(EXPCONF, getdev=True)

    assert len(trainloader)>0, f"trainloader is empty!"
    assert len(devloader)>0, f"devloader is empty!"

    albertconf = MODELCONF[EXPCONF.albert_scale]
    albertconf = modelconf2transformers(albertconf)

    # this is disgraceful.... but just specify things below
    albertconf.vocab_size = len(vocab.itos)
    albertconf.bos_token_id = vocab.stoi['BOS']
    albertconf.eos_token_id = vocab.stoi['EOS']
    albertconf.pad_token_id = vocab.stoi['PAD']



    model = AlbertForPreTraining(albertconf).to(device)

    # huggingface example is doing this for language modeling...
    # https://github.com/huggingface/transformers/blob/v2.6.0/examples/run_language_modeling.py
    no_decay = ['bias', "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": EXPCONF.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(  grouped_parameters,
                        lr = EXPCONF.lr    ) # otherwise, use default
    getsch = get_cosine_schedule_with_warmup if EXPCONF.scheduler =='cosine' else get_linear_schedule_with_warmup
    scheduler = getsch(optimizer, EXPCONF.warmups, EXPCONF.numep*len(trainloader))

    global_step = 0
    L = len(trainloader)
    bsz = len(trainloader[0])

    for ep in tqdm(range(1, EXPCONF.numep+1), desc="epoch progress"):
        lossep_mlm = 0
        lossep_pp = 0
        model.train()
        for i, (b,l,datasetids) in enumerate(tqdm(trainloader, desc="iterations progress"),1):
            '''
            b.input_ids/token_type_ids/attention_mask .shape ==  (bsz, seqmaxlen,)
            b.l.shape == (bsz,)

            ## bert families, when they do MLM with NSP (or other similar sentence based tasks,)
            ## they just uses masked input for their sentence representation encoding, not the unmasked ones
            ## it could be considered as some kind of dropout but at first it looked quite irregular to me.

            ## --> referred to transformers/examples/run_language_modeling.py (v2.1.0)
            ## --> modeling_albert.py ( class AlbertModel.forward() )
            '''

            outputs = model(**b, sentence_order_label=l, return_dict=True )
            global_step+=1

            vsz=outputs.prediction_logits.shape[-1]

            lossmlm = F.cross_entropy(outputs.prediction_logits.detach().view(-1,vsz).contiguous(), b['labels'].view(-1))
            losspp = F.cross_entropy(outputs.sop_logits.detach(), l)

            wandb.log(
                {
                    'step': (i + ep*L)*bsz if EXPCONF.see_bsz_effect else global_step,
                    'train_step/mlm_loss': lossmlm,
                    'train_step/pp_loss': losspp,
                    'train_step/learning_rate': get_lr_from_optim(optimizer),
                }
            )

            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            lossep_mlm += lossmlm
            lossep_pp += losspp

        lossep_mlm/=L
        lossep_pp/=L
        wandb.log(
            {
                'step': ep,
                'train_ep/mlm_loss': lossep_mlm,
                'train_ep/pp_loss': lossep_pp,
            }
        )
        print(f"ep:{ep}: losspp = {lossep_pp}, lossmlm={lossep_mlm}")
        devmlm_loss, devpp_loss = evaldev(EXPCONF, model, devloader, ep)
        savemodel(EXPCONF, model, vocab, ep, mlm=devmlm_loss, pp=devpp_loss)
    return None


def get_arguments_from_cmd(**kwargs):
    for k,v in kwargs.items():
        EXPCONF[k] = v

if __name__ == '__main__':
    #os.environ["WANDB_MODE"] = 'dryrun'
    #os.environ["PYTHONIOENCODING"] = 'utf8'
    Fire(get_arguments_from_cmd)
    if EXPCONF.debug: ## made debug.jsonl by $ head -20 train.jsonl > debugtrain.jsonl etc.
        EXPCONF.bsz = 6
        EXPCONF.numep = 2

    print(EXPCONF)

    wandb.init(project = "scatterlab")
    wandb.config.update(EXPCONF)

    with log_time():
        print("start training")
        main()
