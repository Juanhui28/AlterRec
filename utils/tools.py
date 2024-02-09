"""
Toolkit for data processing and model.

"""

import time
import functools
from datetime import timedelta
import sys, pickle
import json, logging, os, torch, dgl
import logging.config, random
import numpy as np
import torch.nn as nn
 
def log_exec_time(func):
    """wrapper for log the execution time of function
    
    """
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        print('Current Func : {}...'.format(func.__name__))
        start=time.perf_counter()
        res=func(*args,**kwargs)
        end=time.perf_counter()
        print('Func {} took {:.2f}s'.format(func.__name__,(end-start)))
        return res
    return wrapper

def get_time_dif(start_time):
    """calculate the time cost from the start point.
    
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def red_str(s, tofile=False):
    s = str(s)
    if tofile:
        # s = f'**{s}**'
        pass
    else:
        s = f'\033[1;31;40m{s}\033[0m'
    return s

def get_time_str():
    return time.strftime('%Y-%m-%d_%H.%M.%S') + str(time.time() % 1)[1:6]


def get_logger():
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def get_config_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "config")

def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.seed(seed)
    dgl.random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)



def get_full_item_emb(path, model, dataloader, dataloader_tax, dataloader_des, device, epoch, count, args):

    print('get full item emb!')
    tax_embeddings, title_embeddings, des_embeddings = [], [], []

    with torch.no_grad():
        model.eval()
        
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            # outputs = model._first_module().auto_model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs = model.module.model(input_ids=input_ids, attention_mask=attention_mask)
            # import ipdb
            # ipdb.set_trace()
            if args.lm_backbone == 'lora':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']

            

            title_embeddings.append(outputs.detach().cpu())
            torch.cuda.empty_cache()
            

            # print(i)
        title_embeddings = torch.cat(title_embeddings, dim=0)

    
        print('title emb!')

        if dataloader_tax != None:
            for i, batch in enumerate(dataloader_tax):
                input_ids = batch['input_ids'].long().to(device)
                attention_mask = batch['attention_mask'].long().to(device)
            
                if args.lm_backbone == 'lora':
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']


                tax_embeddings.append(outputs.detach().cpu())
                torch.cuda.empty_cache()
            

                # print(i)
            tax_embeddings = torch.cat(tax_embeddings, dim=0)
            # with open(path+'tax_embeddings.pkl','wb') as f:
            #     pickle.dump(tax_embeddings, f)

            print('tax emb!')
        else:
            tax_embeddings = None
        
        for i, batch in enumerate(dataloader_des):
            input_ids = batch['input_ids'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            
            if args.lm_backbone == 'lora':
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']


            des_embeddings.append(outputs.detach().cpu())
            torch.cuda.empty_cache()
            

            # print(i)
        des_embeddings = torch.cat(des_embeddings, dim=0)

        # with open(path+'des_embeddings.pkl','wb') as f:
        #     pickle.dump(des_embeddings, f)
    
        print('des emb!')
        
        if tax_embeddings != None:
            text_embedding = (tax_embeddings + title_embeddings  + des_embeddings)/3

        else:  
            text_embedding = (title_embeddings  + des_embeddings)/2

        with open(path+'/item_embeddings_ep'+str(epoch) + '_iter'+ str(count)+'.pkl','wb') as f:
            pickle.dump(text_embedding, f)

        print('emb saved!')

        

        return text_embedding
    
def adjust_gradient(out_gnn, out_text, label, args):
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    sf_score_gnn = softmax(out_gnn)
    sf_score_text = softmax(out_text)
    score_gnn = sum([sf_score_gnn[i][label[i]] for i in range(out_gnn.size(0))])
    score_text = sum([sf_score_text[i][label[i]] for i in range(out_text.size(0))])

    ratio_gnn = score_gnn / score_text
    ratio_text = 1 / ratio_gnn

    if ratio_gnn > 1:
        coeff_gnn = 1 - tanh(args.alpha * relu(ratio_gnn))
        coeff_text = 1
    else:
        coeff_text = 1 - tanh(args.alpha * relu(ratio_text))
        coeff_gnn = 1

    

    return coeff_gnn, coeff_text
