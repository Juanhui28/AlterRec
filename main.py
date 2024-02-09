import pickle, os
import math
from operator import itemgetter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm
import dgl


from utils.tools import get_logger, get_config_dir, init_seed
from data_loader.data_loader import load_data, construct_graph

import torch.nn.functional as F
from model import AlterRec
import logging
import argparse
from sentence_transformers import SentenceTransformer

from torch.utils.data import Dataset

import copy

log_print = get_logger()

class SessionDataset(Dataset):
    def __init__(self, data,max_item_id, max_len,  max_his, sess2neg=None, mode=None):
        """
        args:
            
            data_type(int): 0: train 1: val 2:test
        """
        super(SessionDataset, self).__init__()

        self.data=data
        self.max_item_id = max_item_id
        # if max_len:
        self.max_seq_len=max_len
        # else:
        #     self.max_seq_len=config['dataset.seq_len']
        self.max_his = max_his
        self.sess2neg = sess2neg
       
        self.mode = mode
    
    def __len__(self):
        return len(self.data)






    def __getitem__(self, index):
        """
        data format:
        <[uid]> <[v1,v2,v3]> <label> <session_id> <[his_s1, his_s2, his_s3]> <auginstance>
        """
     
        
        data=self.data[index]
        uid=torch.tensor([data[0]]) + self.max_item_id+1
       
        label=torch.tensor(data[2])
        
        browsed_ids=np.zeros((self.max_seq_len),dtype=np.int64)
        seq_len=len(data[1][-self.max_seq_len:])
        browsed_ids[:seq_len]=np.array(data[1][-self.max_seq_len:])
        browsed_ids = torch.tensor(browsed_ids)
        # pos_idx=torch.tensor(np.array([seq_len-i-1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int64))

        seq_len=torch.tensor(np.array(seq_len,dtype=np.int64))
        mask=torch.tensor(np.array([1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int64))
        
        if self.mode == 'hard':

            if len(data) == 3:
                aug_hard = torch.tensor(1)
            else: aug_hard = torch.tensor(0)

            aug_hard = aug_hard.bool()



            return uid, browsed_ids, label, mask, aug_hard
           

        return uid, browsed_ids, label, mask
         

def adjust_gradient(model, out_gnn, out_text, label, args):
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

@torch.no_grad()
def get_score(model, x, itemid,mask,max_itemid, device, mode ):
    model.eval()
    if mode == 'gnn':

        ##### no transformer
        # item_emb = x[itemid]

        # seq_emb = torch.sum(mask.unsqueeze(2)*item_emb, 1)
        # seq_emb = seq_emb/(mask.sum(dim=1)).unsqueeze(1)
        # session_emb = seq_emb 
        # item_embs = x[:max_itemid+1]

        # scores_all = torch.matmul(session_emb, item_embs.permute(1, 0))  

        ### use transformer
        scores_all = model.get_text_score(x.to(device),itemid.to(device), mask.to(device), device, max_itemid , mode='gnn')
        scores_all = scores_all.cpu()


    else:
        scores_all = model.get_text_score(x.to(device),itemid.to(device), mask.to(device), device, max_itemid, mode='text' )
        scores_all = scores_all.cpu()

   
    return scores_all






def get_pos(model, data_iter, x, max_itemid,args, device, mode=None, rank_mode=None):

    res50 = []
    res20 = []
    res10 = []
    res5 = []
    labels = []
    new_pos = []
    
    # if not os.path.exists('dataset/user_item_title_des/hard_neg/'+str(mode)+'_sess2neg_'+str(args.start_rank)+'_'+str(args.end_rank)+'.pkl'):
    sess2neg = dict()
    for uid,  browsed_ids, label, mask in tqdm(data_iter):

    
        if mode == 'gnn+text':
            score1 =  get_score(model, x[0].cpu(), browsed_ids,mask,max_itemid, device, mode='gnn' )  ## gnn

            pos_rank = score1.topk(args.end_pos_rank)[1].cpu()[:, args.start_pos_rank:]

        
        else:
            scores = get_score(model, x, browsed_ids,mask,max_itemid, device, mode )
            
            # hard_rank = scores.topk(args.end_rank)[1].cpu()[:,args.start_rank:]
            pos_rank = scores.topk(args.end_pos_rank)[1].cpu()[:, args.start_pos_rank:]

    
        sess = torch.cat([browsed_ids, label], dim=-1)
        for i in range(browsed_ids.size(0)):
           
            if pos_rank != None:
                for j in range(pos_rank.size(1)):
                    # [uid[i],out_seqs[i],label[i], sess, his_sess, auginstance[i]])
                    new_pos.append([uid[i].item()-max_itemid -1,browsed_ids[i].numpy().tolist(), [pos_rank[i][j].item()] ])
                
    

    return  new_pos 
       

        


def get_neg(model, x, browsed_ids,mask,  max_itemid, device, args, mode):

    if mode == 'random':
        hard_rank = torch.randint(0, max_itemid + 1, (browsed_ids.size(0), args.end_rank - args.start_rank))
    
    elif mode == 'gnn+text':
        score1 =  get_score(model, x[0], browsed_ids,mask,max_itemid, device, 'gnn' )  ## gnn
        score2 =  get_score(model, x[1], browsed_ids,mask,max_itemid, device, 'text' )  ## text
        hard_rank1 = score1.topk(args.end_rank)[1].cpu()[:,args.start_rank:]
        hard_rank2 = score2.topk(args.end_rank)[1].cpu()[:,args.start_rank:]

        num_neg = int(hard_rank1.size(1)/2)
        indice1 = torch.multinomial(torch.ones(hard_rank1.size(1)).repeat(hard_rank1.size(0), 1),num_neg, replacement=False  )
        indice2 = torch.multinomial(torch.ones(hard_rank2.size(1)).repeat(hard_rank2.size(0), 1),num_neg, replacement=False  )

        sele_hard_rank1 = torch.gather(hard_rank1, 1, indice1)
        sele_hard_rank2 = torch.gather(hard_rank2, 1, indice2)
        hard_rank = torch.cat([sele_hard_rank1, sele_hard_rank2], dim=-1)
        # pos_rank = score1.topk(args.end_pos_rank)[1].cpu()[:, args.start_pos_rank:]

    elif mode == 'text+random':
        score2 =  get_score(model, x[1], browsed_ids,mask,max_itemid, device, 'text' )  ## text

        hard_rank1 = torch.randint(0, max_itemid + 1, (browsed_ids.size(0), args.end_rank - args.start_rank))
        hard_rank2 = score2.topk(args.end_rank)[1].cpu()[:,args.start_rank:]

        num_neg = int(hard_rank1.size(1)/2)
        indice1 = torch.multinomial(torch.ones(hard_rank1.size(1)).repeat(hard_rank1.size(0), 1),num_neg, replacement=False  )
        indice2 = torch.multinomial(torch.ones(hard_rank2.size(1)).repeat(hard_rank2.size(0), 1),num_neg, replacement=False  )

        sele_hard_rank1 = torch.gather(hard_rank1, 1, indice1)
        sele_hard_rank2 = torch.gather(hard_rank2, 1, indice2)

        hard_rank = torch.cat([sele_hard_rank1, sele_hard_rank2], dim=-1)
        # pos_rank = score1.topk(args

    elif mode == 'gnn+random':
        score2 =  get_score(model, x[0], browsed_ids,mask,max_itemid, device, 'gnn' )

        hard_rank1 = torch.randint(0, max_itemid + 1, (browsed_ids.size(0), args.end_rank - args.start_rank))
        hard_rank2 = score2.topk(args.end_rank)[1].cpu()[:,args.start_rank:]

        num_neg = int(hard_rank1.size(1)/2)
        indice1 = torch.multinomial(torch.ones(hard_rank1.size(1)).repeat(hard_rank1.size(0), 1),num_neg, replacement=False  )
        indice2 = torch.multinomial(torch.ones(hard_rank2.size(1)).repeat(hard_rank2.size(0), 1),num_neg, replacement=False  )

        sele_hard_rank1 = torch.gather(hard_rank1, 1, indice1)
        sele_hard_rank2 = torch.gather(hard_rank2, 1, indice2)

        hard_rank = torch.cat([sele_hard_rank1, sele_hard_rank2], dim=-1)
        # pos_rank = score1.topk(args

    else:
       
        scores = get_score(model, x, browsed_ids, mask, max_itemid, device, mode )
            
        hard_rank = scores.topk(args.end_rank)[1].cpu()[:,args.start_rank:]
    
    return hard_rank


       
def get_neg_mask(new_model, embs, browsed_ids,mask,  max_itemid, device, args,train_mode, label_ori ):
    hard_sam = get_neg( new_model, embs, browsed_ids,mask,  max_itemid, device, args, mode=train_mode)
    hard_neg = torch.cat([label_ori, hard_sam], dim=-1)
    mask_hard = (hard_sam == label_ori)
    mask_hard = torch.cat([torch.zeros(label_ori.size(0)).unsqueeze(1).bool(), mask_hard], dim=-1)

    return hard_neg, mask_hard

            



def train(model, new_model, optimizer, train_iter, x, edge_index, max_itemid, device, scheduler, args,  train_gnn, train_lm, epoch, train_mode, embs):
    model.train()

    res20 = []
    labels = []
    total_loss = total_examples = 0
    total_loss_gnn = total_loss_text = 0
    res20_gnn = []
    res20_text = []

    optimizer1, optimizer2 = optimizer
    scheduler1, scheduler2 = scheduler

    
    i = 0
    # for i, (uid, browsed_ids, label_ori, mask, hard_neg, mask_hard) in enumerate(train_iter):
    for uid, browsed_ids, label_ori, mask, aug_hard in tqdm(train_iter):
        # weights = {name: param for name, param in model.named_parameters()}

        # if train_gnn == 1 and i == 0:
        #     args.id_weight_file.write(str(torch.norm(id_weight).item())  + '\n')
        #     args.id_weight_file.flush()
            
        #     args.text_weight_file.write(str(torch.norm(text_weight).item()) + '\n')
        #     args.text_weight_file.flush()


        optimizer1.zero_grad()
        optimizer2.zero_grad()

        hard_neg_id = torch.tensor(0)
        mask_hard_id = torch.tensor(0)
        hard_neg_text = torch.tensor(0)
        mask_hard_text = torch.tensor(0)

        if args.full_neg:
            hard_neg_id = torch.tensor(0)
            mask_hard_id = torch.tensor(0)
            hard_neg_text = torch.tensor(0)
            mask_hard_text = torch.tensor(0)
        else:
            train_mode_id, train_mode_text = train_mode
            embs_id, embs_text = embs
            # hard_sam = get_neg( new_model, embs, browsed_ids,mask,  max_itemid, device, args, mode=train_mode)
            # hard_neg = torch.cat([label_ori, hard_sam], dim=-1)
            # mask_hard = (hard_sam == label_ori)
            # mask_hard = torch.cat([torch.zeros(label_ori.size(0)).unsqueeze(1).bool(), mask_hard], dim=-1)
            if train_gnn == 1:
                hard_neg_id, mask_hard_id = get_neg_mask(new_model, embs_id, browsed_ids,mask,  max_itemid, device, args,train_mode_id, label_ori )
                
            if train_lm == 1:
                hard_neg_text, mask_hard_text = get_neg_mask(new_model, embs_text, browsed_ids,mask,  max_itemid, device, args,train_mode_text, label_ori )
            

       

        scores, out_gnn, out_text,_ , _, _= model(x.to(device),
                      uid.to(device),
                      edge_index.to(device),
                      max_itemid,
                      browsed_ids.to(device),
                      mask.to(device),
                      hard_neg_id.to(device),
                      mask_hard_id.to(device),
                       hard_neg_text.to(device),
                      mask_hard_text.to(device),
                      train_gnn,
                      train_lm,
                      )
        

        L = nn.CrossEntropyLoss()

        label = torch.zeros(browsed_ids.size(0)).long()
        out_gnn_aug = out_gnn[aug_hard]
        out_text_aug = out_text[aug_hard]
        

        if args.full_neg:
            label_aug = label_ori[aug_hard]
        else:
            label_aug = label[aug_hard]

        no_aug = (1-aug_hard.long()).bool()
        out_gnn_noaug = out_gnn[no_aug]
        out_text_noaug = out_text[no_aug]
        
        if args.full_neg:
            label_noaug = label_ori[no_aug]
        else:
            label_noaug = label[no_aug]

        
        
        if train_gnn == 1:
            if i == 0: print('train gnn')
            # import ipdb
            # ipdb.set_trace()
            loss = L(out_gnn_noaug, label_noaug.to(device).squeeze())

            if aug_hard.sum() > 0:
                if i == 0: print('aug gnn')
                loss_aug =  L(out_gnn_aug, label_aug.to(device).squeeze())
                # if epoch >=2 :
                #     print('gnn loss:', loss)
                #     print('gnn aug loss:', loss_aug)

                loss += args.aug_loss_ratio*loss_aug
            
            
    
                

        if train_lm == 1:
            if i == 0: print('train text')
            loss = L(out_text_noaug, label_noaug.to(device).squeeze())

            if aug_hard.sum() > 0:
                if i == 0: print('aug text')
                loss_aug =  L(out_text_aug, label_aug.to(device).squeeze())
                # if epoch >=2 :
                #     print('text loss:', loss)
                #     print('text aug loss:', loss_aug)
                loss += args.aug_loss_ratio*loss_aug

            

        i += 1


      

        loss.backward()

        if args.modulation !=  'Normal':
            coeff_gnn, coeff_text = adjust_gradient(model, out_gnn, out_text, label, args)
            for name, parms in model.named_parameters():
                
                if 'gcn' in name and parms.grad != None:
                    if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_gnn + \
                                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_gnn
                elif 'linearlayer1' in name and parms.grad != None:
                    if args.modulation == 'OGM_GE':  # bug fixed
                        parms.grad = parms.grad * coeff_text + \
                                        torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                    elif args.modulation == 'OGM':
                        parms.grad *= coeff_text

        

        res20_gnn.append(out_gnn.topk(20)[1].cpu())
        res20_text.append(out_text.topk(20)[1].cpu())

        optimizer1.step()
        optimizer2.step()

        total_loss += loss.item()
        total_examples += label.size(0)

        total_loss_gnn += loss.item()
        total_loss_text += loss.item()

        
        ###
        
        # res20.append(scores.topk(20)[1].cpu())
        # labels.append(label)
        # torch.cuda.empty_cache()

    ###
    # labels = np.concatenate(labels)
    # labels = labels.reshape(-1, 1)
    # acc20, mrr20, ndcg20 = metrics(res20, labels)
    # msg = 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(20, acc20 * 100, mrr20 * 100, ndcg20 * 100)
    
    # # if args.modulation !=  'Nor':
    # acc20, mrr20, ndcg20 = metrics(res20_gnn, labels)
    # msg += 'GNN loss: {},  Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(total_loss_gnn/total_examples, 20, acc20 * 100, mrr20 * 100, ndcg20 * 100)

    # acc20, mrr20, ndcg20 = metrics(res20_text, labels)
    # msg += 'Text loss: {}, Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(total_loss_text/total_examples, 20, acc20 * 100, mrr20 * 100, ndcg20 * 100)
    
    # if train_gnn == 1:
            
    #     with open(args.output_dir+'/id_loss.txt', 'a+') as f:
    #         f.write(str(total_loss_gnn/total_examples)  + '\n')
    #         f.flush()
        
    # if train_lm == 1:
        
    #     with open(args.output_dir+'/text_loss.txt', 'a+') as f:
    #         f.write(str(total_loss_text/total_examples)  + '\n')
    #         f.flush()



    msg = None

    if train_gnn == 1:
        scheduler1.step()
    if train_lm == 1:
        scheduler2.step()
    

    return total_loss / total_examples, msg



def metrics(res, labels):
    res = np.concatenate(res)
    acc_ar = (res == labels)  # [BS, K]
    acc = acc_ar.sum(-1)

    rank = np.argmax(acc_ar, -1) + 1
    mrr = (acc / rank).mean()
    ndcg = (acc / np.log2(rank + 1)).mean()
    return acc.mean(), mrr, ndcg


@torch.no_grad()
def test(model, lm_mdoel, data_iter, edge_index, x, max_itemid, device, args):
    model.eval()
    

    res50 = []
    res20 = []
    res10 = []
    res5 = []
    res1 = []
    res3 = []
    labels = []

    res20_gnn = []
    res20_text = []
    res10_gnn = []
    res10_text = []
    res50_gnn = []
    res50_text = []
    res5_gnn = []
    res5_text = []
    res3_gnn = []
    res3_text = []
    res1_gnn = []
    res1_text = []
    
    fin_score, gnn_score, text_score = [], [], []
    # for i, (uid, browsed_ids, label, mask) in enumerate(data_iter):
    for uid, browsed_ids, label, mask, *aug in tqdm(data_iter):
         
       
        # @profile
        # def get_score(x, uid, edge_index, max_itemid, browsed_ids, mask):
        scores, out_gnn, out_text, _, gnn_emb, lm_emb = model(x.to(device),
                        uid.to(device),
                    edge_index.to(device),
                    max_itemid,
                    browsed_ids.to(device),
                    mask.to(device)
                    )
        torch.cuda.empty_cache()
            # return  scores, out_gnn, out_text, _, gnn_emb, lm_emb 
       
        sub_scores = scores.topk(20)[1].cpu()
        res20.append(sub_scores)
        res10.append(scores.topk(10)[1].cpu())
        res5.append(scores.topk(5)[1].cpu())
        res50.append(scores.topk(50)[1].cpu())
        res1.append(scores.topk(1)[1].cpu())
        res3.append(scores.topk(3)[1].cpu())
        labels.append(label)

        # if args.modulation !=  'Nor':
        res20_gnn.append(out_gnn.topk(20)[1].cpu())
        res10_gnn.append(out_gnn.topk(10)[1].cpu())
        res50_gnn.append(out_gnn.topk(50)[1].cpu())
        res5_gnn.append(out_gnn.topk(5)[1].cpu())
        res1_gnn.append(out_gnn.topk(1)[1].cpu())
        res3_gnn.append(out_gnn.topk(3)[1].cpu())


        res20_text.append(out_text.topk(20)[1].cpu())
        res10_text.append(out_text.topk(10)[1].cpu())
        res50_text.append(out_text.topk(50)[1].cpu())
        res5_text.append(out_text.topk(5)[1].cpu())
        res1_text.append(out_text.topk(1)[1].cpu())
        res3_text.append(out_text.topk(3)[1].cpu())

        if  'amazonm2' not in args.data_name:
            fin_score.append(scores.cpu())
            gnn_score.append(out_gnn.cpu())
            text_score.append(out_text.cpu())
            # return res20, res10, res5, res50, labels, res20_gnn, res10_gnn, res50_gnn, res5_gnn, res20_text, res10_text, res50_text, res5_text, fin_score, gnn_score, text_score

        # scores, out_gnn, out_text, _, gnn_emb, lm_emb  = get_score(x, uid, edge_index, max_itemid, browsed_ids, mask)
        # res20, res10, res5, res50, labels, res20_gnn, res10_gnn, res50_gnn, res5_gnn, res20_text, res10_text, res50_text, res5_text, fin_score, gnn_score, text_score = get_metric(scores, label, out_gnn, out_text)

    # 8450,1,   8450,50
            
    hit20 = []
    labels = np.concatenate(labels)
    labels = labels.reshape(-1, 1)
    acc50, mrr50, ndcg50 = metrics(res50, labels)
    acc20, mrr20, ndcg20 = metrics(res20, labels)
    acc10, mrr10, ndcg10 = metrics(res10, labels)
    acc5, mrr5, ndcg5 = metrics(res5, labels)
    acc1, mrr1, ndcg1 = metrics(res1, labels)
    acc3, mrr3, ndcg3 = metrics(res3, labels)

    # if eval_mode == 'val': print('validation performance')
    # else: print('test performance')
    # print("Top50 : acc {} , mrr {}, ndcg {}".format(acc50, mrr50, ndcg50))
    # print("Top20 : acc {} , mrr {}, ndcg {}".format(acc20, mrr20, ndcg20))
    # print("Top10 : acc {} , mrr {}, ndcg {}".format(acc10, mrr10, ndcg10))
    # print("Top5 : acc {} , mrr {}, ndcg {}".format(acc5, mrr5, ndcg5))

    print('score id norm/lm norm: ', torch.norm(out_gnn, p=2).item(), torch.norm(out_text, p=2).item())
    msg = 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(20, acc20 * 100, mrr20 * 100, ndcg20 * 100)
    msg += 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(1, acc1 * 100, mrr1 * 100, ndcg1 * 100)
    msg += 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format( 3, acc3 * 100, mrr3 * 100, ndcg3 * 100)
    msg += 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)
    msg += 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)
    msg += 'Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format( 50, acc50 * 100, mrr50 * 100, ndcg50 * 100)
   

    metric20 = [acc20, mrr20, ndcg20]
    metric10 = [acc10, mrr10, ndcg10]
    metric5 = [acc5, mrr5, ndcg5]
    metric50  = [acc50, mrr50, ndcg50]
    all_metrics = [metric20, metric10, metric5, metric50]
    hit20.append(acc20)
    # if args.modulation !=  'Nor':
    ########### gnn text performance
    acc50, mrr50, ndcg50 = metrics(res50_gnn, labels)
    acc20, mrr20, ndcg20 = metrics(res20_gnn, labels)
    acc10, mrr10, ndcg10 = metrics(res10_gnn, labels)
    acc5, mrr5, ndcg5 = metrics(res5_gnn, labels)
    acc1, mrr1, ndcg1 = metrics(res1_gnn, labels)
    acc3, mrr3, ndcg3 = metrics(res3_gnn, labels)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(20, acc20 * 100, mrr20 * 100, ndcg20 * 100)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(1, acc1 * 100, mrr1 * 100, ndcg1 * 100)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format( 3, acc3 * 100, mrr3 * 100, ndcg3 * 100)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)
    msg += 'GNN Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(50, acc50 * 100, mrr50 * 100, ndcg50 * 100)
    
    hit20.append(acc20)

    acc50, mrr50, ndcg50 = metrics(res50_text, labels)
    acc5, mrr5, ndcg5 = metrics(res5_text, labels)
    acc20, mrr20, ndcg20 = metrics(res20_text, labels)
    acc10, mrr10, ndcg10 = metrics(res10_text, labels)
    acc1, mrr1, ndcg1 = metrics(res1_text, labels)
    acc3, mrr3, ndcg3 = metrics(res3_text, labels)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(20, acc20 * 100, mrr20 * 100, ndcg20 * 100)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(1, acc1 * 100, mrr1 * 100, ndcg1 * 100)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format( 3, acc3 * 100, mrr3 * 100, ndcg3 * 100)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)
    msg += 'Text Top-{} hit:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(50, acc50 * 100, mrr50 * 100, ndcg50 * 100)
    
    hit20.append(acc20)

    if args.data_name != 'amazonm2':
        score_list = [fin_score, gnn_score,text_score]
    else:
        score_list = [res50, res50_gnn, res50_text]

    item_embs = [gnn_emb, lm_emb]
    return metric20[0], msg, all_metrics, item_embs, score_list, hit20

def output_msg(metric20, metric10, metric5, metric50):

    metric20_mean, metric20_std = metric20.mean(dim=0), metric20.std(dim=0)
    metric10_mean, metric10_std = metric10.mean(dim=0), metric10.std(dim=0)
    metric5_mean, metric5_std = metric5.mean(dim=0), metric5.std(dim=0)
    metric50_mean, metric50_std = metric50.mean(dim=0), metric50.std(dim=0)

    msg = 'Top-{} hit:{:.2f} ± {:.2f}, mrr:{:.2f} ± {:.2f},  ndcg:{:.2f} ± {:.2f} \n'.format(20, metric20_mean[0] * 100, metric20_std[0]*100 , metric20_mean[1] * 100,  metric20_std[1]*100,   metric20_mean[2] * 100, metric20_std[2]*100 )
    msg += 'Top-{} hit:{:.2f} ± {:.2f}, mrr:{:.2f} ± {:.2f},  ndcg:{:.2f} ± {:.2f}\n'.format(10, metric10_mean[0] * 100, metric10_std[0]*100 , metric10_mean[1] * 100,  metric10_std[1]*100,   metric10_mean[2] * 100, metric10_std[2]*100 )
    msg += 'Top-{} hit:{:.2f} ± {:.2f}, mrr:{:.2f} ± {:.2f},  ndcg:{:.2f} ± {:.2f} \n'.format(5, metric5_mean[0] * 100, metric5_std[0]*100 , metric5_mean[1] * 100,  metric5_std[1]*100,   metric5_mean[2] * 100, metric5_std[2]*100 )
    msg += 'Top-{} hit:{:.2f} ± {:.2f}, mrr:{:.2f} ± {:.2f},  ndcg:{:.2f} ± {:.2f} \n'.format( 50, metric50_mean[0] * 100, metric50_std[0]*100 , metric50_mean[1] * 100,  metric50_std[1]*100,   metric50_mean[2] * 100, metric50_std[2]*100 )
    
    return msg

def get_last_epoch(model):
    new_model = copy.deepcopy(model)

    for param in new_model.parameters():
        param.requires_grad=False

    return new_model


def main():
    parser = argparse.ArgumentParser(description='sessionRec')
    parser.add_argument('--data_name', type=str, default='user_item_title_des') ##choices='amazonm2/sess_UK_text_UK_full/, amazonm2/sess_ES_text_ES/, user_item_title_des'
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='./dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512*16)
    parser.add_argument('--epoch', type=int, default=20) 
    parser.add_argument('--id_hidden_channel', type=int, default=128)
    parser.add_argument('--lm_hidden_channel', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--kill_cnt', type=int, default=15)
    parser.add_argument('--lr_dc_step_id', type=int, default=3)
    parser.add_argument('--lr_dc_step_text', type=int, default=3)
    parser.add_argument('--lr_dc', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=10)
    parser.add_argument('--max_his', type=int, default=10)
    parser.add_argument('--sample_size', type=int, default=12)

    parser.add_argument('--text_input_gnn', action='store_true', help='use the text embedding as the input to the gnn')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--finetune', action='store_true')

    ### adjust gradients:
    parser.add_argument('--alpha',  type=float,default=0.1, help='alpha in OGM-GE')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--lr_id', type=float, default=0.001)
    parser.add_argument('--lr_text', type=float, default=0.001)

    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--transformer_block", type=int, default=2)
    # parser.add_argument('--text_input_gnn', action='store_true', default=True, help='use the text embedding as the input to the gnn')
    # parser.add_argument('--finetune', action='store_true', default=True,)
    parser.add_argument('--full_neg', action='store_true', help='use the full neg or not')
    parser.add_argument('--only_train_id', action='store_true')
    parser.add_argument('--only_train_text', action='store_true')

    ### iterative train
    parser.add_argument("--end_rank", type=int, default=5000)
    parser.add_argument("--start_rank", type=int, default=50)
    parser.add_argument("--start_pos_rank", type=int, default=1)
    parser.add_argument("--end_pos_rank", type=int, default=5)
    parser.add_argument("--aug_loss_ratio", type=float, default=0.1)
    parser.add_argument("--gap_epoch", type=int, default=2)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.5)


    #### adaptor
    parser.add_argument("--n_exps", type=int, default=8)
    parser.add_argument("--adaptor_layers", type=list, default=[512,300])
    parser.add_argument("--adaptor_dropout_prob", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument('--id_module', type=str, default='average') ### transformer


    
    
    args = parser.parse_args()
    print(args)

   

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    print('current dataset:', args.data_name)
    train_data, val_data, test_data, max_userid, max_itemid, max_sessionid, item_prob_list = load_data(args.data_name, args.input_dir)
    
    print('max item: ', max_itemid)
    path = args.input_dir +'/' + args.data_name

   
    if 'amazon' in args.data_name:

        ####### sentencebert
        # path = path + '/' + 'sbert/lastlayer_finetune_text_embeddings'
        ### sbert
        # path = path + '/' + 'sbert/text_embeddings'   
        #### mt5
        # path = path + '/' + 'mt5/text_embeddings'

        ### stsb-xlm-r-multilingual
        # path = path + '/' + 'stsb/text_embeddings'

        if 'UK' in args.data_name:
            path = path + '/' + 'all_mini/text_embeddings'

        else:
            path = path + '/' + 'dist/text_embeddings'


        
        if os.path.exists(path+'/tax_embeddings.pkl'):
            with open(path+'/tax_embeddings.pkl','rb') as f:
                
                tax_embedding =pickle.load(f)
            
        else:
            tax_embedding = None

        with open(path+'/title_embeddings.pkl','rb') as f:
            
            title_embedding=pickle.load(f)
        
        

        with open(path+'/des_embeddings.pkl','rb') as f:
            des_embedding=pickle.load(f)
        
        
        
        print(path)
        print('load emebddings from fixed model !!')
        if tax_embedding != None:
            title_embedding = (tax_embedding + title_embedding + des_embedding)/3
        else:
            title_embedding = (title_embedding + des_embedding)/2
    else:
    ####### finetune sentencebert
       
        embedding_path = 'output/0923/sbert_lastlayer_finetune/item_embeddings_ep12_iter0.pkl'
    
        with open(embedding_path,'rb') as f:
            title_embedding = pickle.load(f)
        
        print(embedding_path)
        print('load emebddings!!')

    

    


    print('The size of train data:', len(train_data))
    print('The size of valid data',len(val_data))
    print('The size of test data', len(test_data))
    
    # train_edge = construct_graph(args.data_name, args.input_dir, max_itemid, args.sample_size)
    train_edge, G = construct_graph(args.data_name,  args.sample_size, args.input_dir,max_itemid)
   
    if 'amazonm2' in args.data_name :
        totle_node_num =   max_itemid + 1
    else:
        totle_node_num = max_userid + 1 + max_itemid + 1
    item_num =  max_itemid + 1
    print('total number in gnn', totle_node_num)
    
    test_data = SessionDataset(test_data, max_itemid, args.max_len,  args.max_his)
    val_data = SessionDataset(val_data, max_itemid, args.max_len,  args.max_his)
    train_data_sess = SessionDataset(train_data, max_itemid, args.max_len, args.max_his)


    train_iter = DataLoader(dataset=train_data_sess,
                            batch_size=args.batch_size,
                            num_workers=4,##4
                            shuffle=True,
                            pin_memory=False,
                            # collate_fn=SessionDataset.collate_fn
                            )
    

    test_iter = DataLoader(dataset=test_data,
                        batch_size=args.test_batch_size,
                        num_workers=4, ##4
                        shuffle=False,
                        pin_memory=False,
                        # collate_fn=SessionDataset.collate_fn
                        )
    
    val_iter = DataLoader(dataset=val_data,
                        batch_size=args.test_batch_size,
                        num_workers=4, ##4
                        shuffle=False,
                        pin_memory=False,
                        # collate_fn=SessionDataset.collate_fn
                       )
    

    metric20_val, metric10_val, metric5_val, metric50_val = [], [], [], []
    metric20_test, metric10_test, metric5_test, metric50_test = [], [], [], []

    # hit20_total_out = open(args.output_dir+'/test_total_hit20.txt', 'w+')
    # hit20_id_out = open(args.output_dir+'/test_id_hit20.txt', 'w+')
    # hit20_text_out = open(args.output_dir+'/test_text_hit20.txt', 'w+')
    # args.id_weight_file = open(args.output_dir+'/id_model_norm.txt', 'w+')
                
    # args.text_weight_file = open(args.output_dir+'/text_model_norm.txt', 'w+')
    
    # args.id_loss_file = open(args.output_dir+'/id_loss.txt', 'w+') 
       
    # args.text_loss_file =  open(args.output_dir+'/text_loss.txt', 'w+') 
    
    for _ in range(1):

        best_hit = 0
        best_msg = ''
        kill = 0
        best_val_meg = ''
        best_metric_val =  0
        best_metric_test =  0

        seed = args.seed
        init_seed(seed)
        print('seed: ', seed)
        train_gnn =1
        train_lm = 0
        
        model = His_GNN(totle_node_num, item_num, args.id_hidden_channel, args.lm_hidden_channel, args.id_hidden_channel, args.num_layers, args.dropout,args, title_embedding, item_prob_list, train_edge)
       
        # else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
        optimizer1 = torch.optim.Adam(model.id_module.parameters(), lr=args.lr_id, weight_decay=args.l2)
        optimizer2 = torch.optim.Adam(model.text_module.parameters(), lr=args.lr_text, weight_decay=args.l2)
        # optimizer = torch.optim.Adam([{'params': model.id_module.parameters(), 'lr': args.lr_gnn}, 
        #                              {'params': model.text_module.parameters(), 'lr': args.lr_text}], 
        #                              weight_decay=args.l2)

        lm_model = None

        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=args.lr_dc_step_id, gamma=args.lr_dc)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.lr_dc_step_text, gamma=args.lr_dc)

        optimizer = [optimizer1, optimizer2]
        scheduler = [scheduler1, scheduler2]

        model.reset_parameters()

        
        model = model.to(device)
        
        print('start training, seed: ', seed)

     
        item_text_embed = None
    
        # sess2neg_gnn, _ = get_hard_neg(model, train_iter, item_text_embed, max_itemid, args, device, mode='gnn', rank_mode='ori') 
        
        train_data_sess = SessionDataset(train_data, max_itemid, args.max_len, args.max_his , mode='hard')
        train_hard_iter = DataLoader(dataset=train_data_sess,
                        batch_size=args.batch_size,
                        num_workers=4,##4
                        shuffle=True,
                        pin_memory=False,
                        # collate_fn=SessionDataset.collate_fn
                    )
        
        if args.only_train_id:
            train_gnn = 1
            train_lm=0
        elif args.only_train_text:
            train_gnn = 0
            train_lm=1
        else:
            train_gnn = 1
            train_lm=0



        train_mode_id = train_mode_text  = 'random'
        embs_id = None
        embs_text = None

        train_mode = [train_mode_id, train_mode_text]
        embs = [embs_id, embs_text]

        new_model = get_last_epoch(model)

        for epoch in range(args.epoch):
               
            # if  epoch >= args.gap_epoch and (not args.only_train_id) and (not args.only_train_text) : 
            if  train_gnn == 1  and epoch > 0 and (not args.only_train_id) and (not args.only_train_text) and epoch % 2 == 0:
                    
                item_gnn_embed = item_embs[0].cpu()
                new_model = get_last_epoch(model)

                train_gnn = 0
                train_lm = 1
                

                if epoch >= args.gap_epoch:
                    # new_pos_sample = get_pos(model, train_iter,  item_embs[1].cpu(), max_itemid, args, device, mode='text')
                
                    # new_pos_sample  = get_pos(new_model, train_iter, item_gnn_embed, max_itemid, args, device, mode='gnn')
                

                    # print('get hard neg from gnn. train text!!!!')
                    # train_data_new = train_data + new_pos_sample
                    # train_data_sess = SessionDataset(train_data_new, max_itemid, args.max_len, args.max_his, mode='hard')
                    # train_hard_iter = DataLoader(dataset=train_data_sess,
                    #                 batch_size=args.batch_size,
                    #                 num_workers=4,##4
                    #                 shuffle=True,
                    #                 pin_memory=False,
                    #                 # collate_fn=SessionDataset.collate_fn
                    #             )
                    
                    train_mode_text = 'gnn+random'
                    embs_text = item_embs

                    # train_mode_text = 'gnn'
                    # embs_text = item_gnn_embed

                else:
                    train_mode_text = 'random'
                
            elif train_lm == 1 and epoch > 0 and  (not args.only_train_id) and (not args.only_train_text) and epoch % 2 == 0:
                
                new_model = get_last_epoch(model)

                train_gnn = 1
                train_lm = 0

                if epoch >= args.gap_epoch:

                    # new_pos_sample = get_pos(model, train_iter,  item_embs[1].cpu(), max_itemid, args, device, mode='text')
                
                    # new_pos_sample = get_pos(new_model, train_iter, item_embs[0].cpu(), max_itemid, args, device, mode='gnn')
                    
                    
                    # print('get hard neg from text. train gnn!!!!')
                    # train_data_new = train_data + new_pos_sample
                    # train_data_sess = SessionDataset(train_data_new, max_itemid, args.max_len, args.max_his, mode='hard')
                    # train_hard_iter = DataLoader(dataset=train_data_sess,
                    #                 batch_size=args.batch_size,
                    #                 num_workers=4,##4
                    #                 shuffle=True,
                    #                 pin_memory=False,
                    #                 # collate_fn=SessionDataset.collate_fn
                    #             )
                
                    # train_mode_id = 'gnn+text'
                    # embs_id = item_embs
                
                    train_mode_id = 'text+random'
                    embs_id = item_embs

                    # train_mode_id = 'gnn+random'
                    # embs_id = item_embs


                    # train_mode_id = 'gnn'
                    # embs_id =  item_embs[0].cpu()

                    # train_mode_id = 'text'
                    # embs_id =  item_embs[1].cpu()

                    # train_mode_id = 'random'

                else:
                    train_mode_id = 'random'

            train_mode = [train_mode_id, train_mode_text]
            embs = [embs_id, embs_text]

           

            loss, _ = train(model,new_model, optimizer, train_hard_iter, title_embedding,  train_edge, max_itemid, device, scheduler, args, train_gnn, train_lm, epoch, train_mode=train_mode, embs=embs)

            # model.load_state_dict(torch.load('output/tmp/model_lr0.01_dp0.1_dim128.bin',  map_location='cpu'))
            
            # train_hit, train_info, metrics_train, _, _, _ = test(model, lm_model, train_hard_iter, train_edge,title_embedding, max_itemid, device, args)


            val_hit, val_info, metrics_val, _, _ , _= test(model, lm_model, val_iter, train_edge,title_embedding, max_itemid, device, args)

            test_hit, test_info, metrics_test, item_embs, score_list, hit20_list = test(model, lm_model, test_iter, train_edge, title_embedding, max_itemid, device, args)

            # with open(args.output_dir+'/test_total_hit20.txt', 'a+') as f:
            #     f.write(str(round(hit20_list[0],4).item()) + '\n')
            #     f.flush()
            
            # with open(args.output_dir+'/test_id_hit20.txt', 'a+') as f:
            #     f.write(str(round(hit20_list[1],4).item()) + '\n')
            #     f.flush()
            
            # with open(args.output_dir+'/test_text_hit20.txt', 'a+') as f:
            #     f.write(str(round(hit20_list[2],4).item()) + '\n')
            #     f.flush()
            
            # msg = f'run [{seed}] epoch[{epoch}]  val :{val_info}'
            # log_print.info(msg)

            # msg = f'run [{seed}] epoch[{epoch}] test :{test_info}'
            # log_print.info(msg)

            
            # msg = f'run [{seed}] epoch[{epoch}] loss {loss} train :{train_info}'
            # log_print.info(msg)

            
            msg = f'run [{seed}] epoch[{epoch}] loss {loss} val :{val_info}'
            log_print.info(msg)

            msg = f'run [{seed}] epoch[{epoch}] loss {loss} test :{test_info}'
            log_print.info(msg)

            
        
            
            if val_hit > best_hit:
                best_hit = val_hit
                msg = f'run [{seed}] epoch[{epoch}] loss {loss} test :{test_info}'
                best_msg = msg
                best_val_meg = f'run [{seed}] epoch[{epoch}] loss {loss} val :{val_info}'

                best_metric_val =  metrics_val
                best_metric_test =  metrics_test

                kill=0
                if args.save:
                    torch.save(model.state_dict(), args.output_dir+'/model_lr'+str(args.lr)+'_dp'+str(args.dropout)+'_dim'+str(args.id_hidden_channel)+'.bin')
                    torch.save(item_embs, args.output_dir+'/item_embs_lr'+str(args.lr)+'_dp'+str(args.dropout)+'_dim'+str(args.id_hidden_channel)) 
                    torch.save(score_list, args.output_dir+'/scores_lr'+str(args.lr)+'_dp'+str(args.dropout)+'_dim'+str(args.id_hidden_channel)) 

            else:
                # msg = f'epoch[{epoch}] test :{test_info}'
                
                # log_print.info(msg)
                kill += 1
                if kill >= args.kill_cnt:
                    log_print.info('Early stop: No more improvement')
                    # log_print.info(best_val_meg)
                    # log_print.info(best_msg)
                    
                    break
            log_print.info(best_val_meg)
            log_print.info(best_msg)
            
        log_print.info('the best performance:')
        log_print.info(best_val_meg)
        log_print.info(best_msg)

        # metrics = [metric20, metric10, metric5, metric50]
        metric20_val.append(best_metric_val[0])
        metric10_val.append(best_metric_val[1])
        metric5_val.append(best_metric_val[2])
        metric50_val.append(best_metric_val[3])

        metric20_test.append(best_metric_test[0])
        metric10_test.append(best_metric_test[1])
        metric5_test.append(best_metric_test[2])
        metric50_test.append(best_metric_test[3])

    metric20_val, metric10_val, metric5_val, metric50_val = torch.tensor(metric20_val), torch.tensor(metric10_val), torch.tensor(metric5_val), torch.tensor(metric50_val)
    metric20_test, metric10_test, metric5_test, metric50_test = torch.tensor(metric20_test), torch.tensor(metric10_test), torch.tensor(metric5_test), torch.tensor(metric50_test)

    msg_val = output_msg(metric20_val, metric10_val, metric5_val, metric50_val)
    msg_test = output_msg(metric20_test, metric10_test, metric5_test, metric50_test)
    print('valid performance:')
    print(msg_val)
    # print('\n')
    print('test performance:')
    print(msg_test)
    


    


 

if __name__ == "__main__":
    main()


