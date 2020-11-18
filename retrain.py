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


class MLP(nn.Module):
    def __init__(self, expconf, insize, outsize, morelayers=0):
        super().__init__()
        self.expconf = expconf
        layers = [nn.Linear(insize, insize//2), nn.Linear(insize//2, insize//4)]
        for i in range(morelayers):
            layers.append(nn.Linear(insize//4, insize//4))

        layers.append(nn.Linear(insize//4, outsize))
        self.layers = nn.Sequential( *layers
                                    )

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(F.dropout(layer(x), p=self.expconf.cls_do_p))
        return x

def accuracy(soplogits, soplabels):
    return (soplogits.argmax(dim=1) == soplabels).float().sum().item() / len(soplabels)

def evaldev(expconf, albertmodel, clsmodel, devloader, global_step, infernow=False):
    if not infernow:
        clsmodel.eval()
        L = len(devloader)
        bsz= len(devloader[0])

        losspp = 0
        acc = 0


        for i, (b, l, datasetids) in enumerate(tqdm(devloader, desc="eval iter progress")):
            outputs = albertmodel(**b, return_dict=True)
            logits = clsmodel(outputs.pooler_output)

            losspp += F.cross_entropy(logits, l).item()
            acc += accuracy(logits, l)


    else: #infernow:
        albertmodel.eval()
        L = len(devloader)
        bsz= len(devloader[0])

        #lossmlm = 0
        losspp = 0
        acc = 0


        for i, (b, l, datasetids) in enumerate(tqdm(devloader, desc="eval iter progress")):
            outputs = albertmodel(**b, sentence_order_label=l, return_dict=True)
            #vsz= outputs.prediction_logits.shape[-1]
            #lossmlm += F.cross_entropy(outputs.prediction_logits.detach().view(-1,vsz).contiguous(), b['labels'].view(-1)).item()
            losspp += F.cross_entropy(outputs.sop_logits, l).item()
            acc += accuracy(outputs.sop_logits, l)

    #lossmlm /= L
    losspp /= L
    acc /= L

    wandb.log(
    {
            #'dev/mlm_loss': lossmlm,
            'dev/pp_loss': losspp,
            'dev/pp_acc': acc,
    } )

    return losspp, acc


def write_sub(expconf, albert, cls, global_step, acc=0., testloader=None, infernow=False):
    savedir = Path(expconf.modelsaveroot) / (get_date() + '-cls')
    if not savedir.is_dir():
        Path.mkdir(savedir, parents=True)
    expconf.model_date_name = Path(expconf.model_date_name)
    loadedalbert_date, loadedalbert_name = expconf.model_date_name.parent.name, expconf.model_date_name.name
    savename = f"sub_mlp{acc:.3f}_{get_time()}__{loadedalbert_date}.{loadedalbert_name}"
    if expconf.infer_now:
        savename = f"infer_now_{get_time()}_{loadedalbert_date}.{loadedalbert_name}"
    savepath = savedir/savename
    if expconf.debug:
        savepath = Path('./testinfer.out')
    with savepath.open(mode= 'w') as f:
        for b, _, datasetids in tqdm(testloader, desc=f"MLP{acc:.3f}: writing submission file"):
            if not expconf.infer_now:
                outputs = albert(**b, return_dict=True)
                logits = cls(outputs.pooler_output)
            else: #infernow
                outputs = albert(**b, return_dict=True)
                logits = outputs.sop_logits
            inferred = logits.argmax(dim=1).long().tolist() if not logits.dim()==1 else logits.argmax(dim=0).tolist()
            for id,ans in zip(datasetids, inferred):
                line = f"{id},{ans}\n"
                f.write(line)
    print(str(savepath))

    return None

#def savemodel(expconf, model, vocab, global_step, acc=0):
def savemodel(expconf, albert, cls, vocab, global_step, acc=0.):
    d_expconf = expconf.toDict()
    saveroot = Path(expconf.modelsaveroot)
    todaydir = saveroot / (get_date() + '-cls')
    if not todaydir.is_dir():
        Path.mkdir(todaydir, parents=True)

    savename = f"MLP_{acc:.3f}_{get_time()}_step{global_step}.lr{expconf.lr}.w{expconf.cls_warmups}.sch{expconf.cls_sch}.bsz{expconf.bsz}.pth"
    saved = dict()
    saved = {
        'expconf': d_expconf,
        'albert': albert.state_dict(),
        'model': cls.state_dict(),
        'vocab': vocab
    }
    savepath = todaydir/savename
    print(f"saving {savename}\n\tat {str(todaydir)}")
    torch.save(saved, savepath)

def loadmodel_info(expconf):
    root = Path(expconf.modelsaveroot)
    date_name  = expconf.model_date_name # ==  11-xx/*.pth
    loaded = Munch(torch.load(root/date_name, map_location=torch.device('cpu')) )
    vocab = loaded.vocab
    model_weight = loaded.model
    trained_condition = loaded.expconf
    trained_condition = Munch(trained_condition)

    return model_weight, vocab, trained_condition

def retrieve_conf(trained_condition, trained_vocab):
    albertconf = AlbertConfig.from_pretrained(f'albert-{trained_condition.albert_scale}-v2')
    if 'smaller' in trained_condition.keys():
        if trained_condition.smaller: #originally used 4H for FFN but for memory issue, use 1H for FFN
            albertconf.hidden_size = trained_condition.hidden_size
            albertconf.num_hidden_layers = trained_condition.num_hidden_layers
            albertconf.num_attention_heads = trained_condition.num_attention_heads
            albertconf.intermediate_size = albertconf.hidden_size

    albertconf.vocab_size = len(trained_vocab.itos)
    albertconf.bos_token_id = trained_vocab.stoi['BOS']
    albertconf.eos_token_id = trained_vocab.stoi['EOS']
    albertconf.pad_token_id = trained_vocab.stoi['PAD']
    albertconf.max_position_embeddings = 40

    return albertconf

def main():
    # my dice shows 777 only. period.
    random.seed(EXPCONF.seed)
    np.random.seed(EXPCONF.seed)
    torch.manual_seed(EXPCONF.seed)
    torch.cuda.manual_seed_all(EXPCONF.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tempconf = EXPCONF.copy()
    tempconf.datamode = 'test'

    testloader, ___, _____  = get_loader(tempconf)
    trainloader, __, _trainds = get_loader(EXPCONF, getdev=False)
    devloader, _, _devds = get_loader(EXPCONF, getdev=True)

    assert len(trainloader)>0, f"trainloader is empty!"
    assert len(devloader)>0, f"devloader is empty!"

    # this is disgraceful.... but just specify things below
    model_weight, vocab, trained_condition = loadmodel_info(EXPCONF)

    albertconf = retrieve_conf(trained_condition, vocab)
    albert = AlbertForPreTraining(albertconf)
    albert.load_state_dict(model_weight)
    albert=albert.to(device)

    global_step = 0
    L = len(trainloader)
    bsz = len(trainloader[0])

    if not EXPCONF.infer_now:
        albert=albert.albert
        albert.eval() # freeze

        cls = MLP(EXPCONF, albertconf.hidden_size, 2).to(device)
        cls.train()
        for p in cls.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # huggingface example is doing this for language modeling...
        # https://github.com/huggingface/transformers/blob/v2.6.0/examples/run_language_modeling.py
        optimizer = AdamW(  cls.parameters(),
                            lr = EXPCONF.cls_lr    ) # otherwise, use default
        getsch = get_cosine_schedule_with_warmup if EXPCONF.cls_sch =='cosine' else get_linear_schedule_with_warmup
        scheduler = getsch(optimizer, EXPCONF.cls_warmups, EXPCONF.cls_numsteps)

        ## train cls only!
        while global_step < EXPCONF.cls_numsteps:
            lossep_pp = 0
            accep_pp = 0
            cls.train()
            for i, (b,l,datasetids) in enumerate(tqdm(trainloader, desc="iterations progress"),1):
                outputs = albert(**b, return_dict=True )
                global_step+=1

                logits = cls(outputs.pooler_output)
                losspp = F.cross_entropy(logits, l)

                lossppval = losspp.item()
                acc = accuracy(logits.clone().detach(), l)

                wandb.log(
                    {
                        'step': global_step,
                        'cls.train_step/learning_rate': get_lr_from_optim(optimizer),
                        'cls.train_step/pp_loss': lossppval,
                        'cls.train_step/pp_acc': acc,
                    }
                )

                optimizer.step()
                scheduler.step()
                cls.zero_grad()

                lossep_pp += lossppval
                accep_pp += acc
                if global_step%EXPCONF.logevery==0:
                    lossep_pp/=L
                    accep_pp/=L

                    wandb.log(
                        {
                            'cls.train_ep/pp_loss': lossep_pp,
                            'cls.train_ep/pp_acc': accep_pp,
                        }
                    )
                    devpp_loss, devpp_acc = evaldev(EXPCONF, albert, cls, devloader, global_step)
                    if devpp_acc > EXPCONF.savethld:
                        savemodel(EXPCONF, albert, cls, vocab, global_step, acc=devpp_acc)
                        write_sub(EXPCONF, albert, cls, global_step, acc=devpp_acc, testloader= testloader)

    else: # infer now
        cls= None
        devpp_loss, devpp_acc = evaldev(EXPCONF, albert, cls, devloader, global_step, infernow= EXPCONF.infer_now)
        write_sub(EXPCONF, albert, cls, global_step, acc=devpp_acc, testloader= testloader, infernow= EXPCONF.infer_now)




    return None


def get_arguments_from_cmd(**kwargs):
    for k,v in kwargs.items():
        EXPCONF[k] = v


if __name__ == '__main__':
    #os.environ["WANDB_MODE"] = 'dryrun'
    #os.environ["PYTHONIOENCODING"] = 'utf8'
    Fire(get_arguments_from_cmd)
    EXPCONF.clstrain = True #when running this, clstrain == True always
    if EXPCONF.debug: ## made debug.jsonl by $ head -20 train.jsonl > debugtrain.jsonl etc.
        EXPCONF.bsz = 6
        EXPCONF.numep = 2
        EXPCONF.warmups = 3
        EXPCONF.alpha_warmup = True
        EXPCONF.cls_numsteps = 30
        EXPCONF.logevery=10

    wandb.init(project = "MLP_albert")
    wandb.config.update(EXPCONF)

    with log_time():
        print("retrain with trained weight")
        main()
