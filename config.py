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
    #'use_pretrained': False,
    'smaller': True,
        'hidden_size': 128,
        'num_hidden_layers': 8,
        'num_attention_heads': 8,

    # datapath and dataloader  == loading f"{dataroot}/{mode}{kfold_k}.jsonl"
    'dataroot': 'data/',
    'datamode':'train', # train, dev, test
    'kfold_k': 0, # set the split you want
    'vocabpath': 'data/vocab.json',
    'numworkers': 0, #hard to tell what is optimal... but consider number of cpus we have https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5

    'see_bsz_effect': False, #with this option =True, logs are recorded with x = number of examples seen
                            # this is just confusing...

    # training conditions
    'numep': 10, # later optimize
    'bsz': 512,
    'scheduler': 'cosine', # linear
    'warmups': 100,
    'lr': 1e-4,
    'modelsaveroot': 'model/', #path to save .pth
    # PP loss balancing coeff  alpha_pp
    'alpha_pp': 0.5, # float
        'alpha_warmup':False, # if True, it grows from 0 to alpha_pp according to warmup_steps

    #adamW
    'weight_decay': 0,


    # experiment condition
    'maskratio': 0.15,
    'masking': 'random', # span (span masking used for ALBERT original paper )
        'span_n': 3, # to what n-gram would span masking cover
    'savethld':0.45,

    ### classification retrain.py configs
    'cls_morelayers':0,
    'clstrain': False,
    'cls_lr': 1e-4,
    'cls_do_p': 0.1,
    'cls_numsteps': 2000,
    'cls_sch': 'linear', #cosine
    'cls_warmups': 200,
    'logevery': 100,
    'model_date_name': 'date/modelpathtobeloaded', # modelsaveroot/date/name
    ## immediate inference
    'infer_now': False,

}

EXPCONF = Munch(EXPCONF)
