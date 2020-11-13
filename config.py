from munch import Munch

#from .data.prepdata import CLS, SEP, PAD, MASK
# according to the csv, max(tokens) = 30005
# while, they are sparse (len(set(tokens)) == ~8000, not 30005), so consider constracting it if need more expansion of scale in memory


## this is for experiment configuration
EXPCONF = {
    #debug option
    'debug':False,
    #role some dices
    'seed': 777,

    # model scaling
    'albert_scale' : 'base', # base, xlarge
    'use_pretrained': False,
    'ffndim2hidden': False,

    # datapath and dataloader  == loading f"{dataroot}/{mode}{kfold_k}.jsonl"
    'dataroot': 'data/',
    'datamode':'train', # train, dev, test
    'kfold_k': 0, # set the split you want
    'vocabpath': 'data/vocab.json',
    'numworkers': 0, #hard to tell what is optimal... but consider number of cpus we have https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5

    'see_bsz_effect': True, #with this option =True, logs are recorded with x = number of examples seen

    # training conditions
    'numep': 10, # later optimize
    'bsz': 512,
    'scheduler': 'cosine', # linear
    'warmups': 100,
    'lr': 1e-4,
    'modelsaveroot': 'model/', #path to save .pth

    #adamW
    'weight_decay': 0.01,


    # for later inference
    'modelloadpath': 'model/tobeloaded.pth',

    # experiment condition
    'maskratio': 0.15,
    'masking': 'random', # span (span masking used for ALBERT original paper )
        'span_n': 3, # to what n-gram would span masking cover


}

EXPCONF = Munch(EXPCONF)
