import json
from contextlib import contextmanager
from time import time
from pathlib import Path
from datetime import datetime
from munch import Munch

def open_vocab(vocabpath):
    vocab= Munch(json.load(open(vocabpath)))
    return vocab

def get_lr_from_optim(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
        break
    return lr

def get_date():
    now = datetime.now()
    return now.strftime('%m-%d')
def get_time():
    now = datetime.now()
    return now.strftime('%H.%M.%S')



@contextmanager
def log_time():
    start = time()
    yield
    s = time()-start
    m = s/60
    h = m/60

    print(f"spent {str(h)[:4]} hrs!\n")
    print(f"=spent {str(m)[:4]} mins!\n")
    print(f"=spent {str(s)[:4]} secs!\n")

    return None
