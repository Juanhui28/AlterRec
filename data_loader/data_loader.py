
import os
import pickle
import dgl
import json
from multiprocessing import Pool
from multiprocessing import cpu_count 
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_geometric.utils import coalesce


def common_seq(data_list, max_itemid):
    out_seqs=[]
    label=[]
    uid=[]
    auginstance = []
    sessionid = []
    count = 0

    for u in tqdm(data_list,desc='gen_seq...',leave=False):
        
        u_seqs=data_list[u]
        for sess, seq in u_seqs.items():      
            for i in range(1,len(seq)):
            # for i in range(1,2): ######### make change here to test for session similarity
                uid.append(int(u))
                out_seqs.append(seq[:-i])
                label.append([seq[-i]])
                if i == 1: 
                    auginstance.append(sess)
                else:
                    auginstance.append(-1)
                sessionid.append(sess)
    final_seqs=[]
    for i in range(len(uid)):
        sess = sessionid[i]
        userid = uid[i]
       
        final_seqs.append([uid[i],out_seqs[i],label[i], sess, auginstance[i]])

    return final_seqs

def max_num(data):
    max_vid=0
    max_uid=0
    max_seesionid = 0
    for u in data:
        if u>max_uid:
            max_uid=u
        for sess, item in data[u].items():
            if max_vid<max(item):
                max_vid=max(item)
            if sess > max_seesionid: max_seesionid = sess
    
    return max_uid, max_vid, max_seesionid

def get_item_prob_list(data_list, max_itemid):

    train_item_counts =  [0] * (max_itemid + 1)
    for data in data_list:
        for item in data[1]:
            train_item_counts[item] += 1
    
    pop_prob_list = np.power(train_item_counts, 1.0)
   
    pop_prob_list = pop_prob_list / sum(np.array(pop_prob_list))
    return torch.tensor(pop_prob_list)




def get_max_from_seq(data_list):
    max_itemid=0
    max_userid=0

    max_seesionid = 0
    for data in data_list:
        if data[0]>max_userid:
            max_userid=data[0]
        if max_itemid<max(data[1]):
            max_itemid=max(data[1])
        if max_itemid<max(data[2]):
            max_itemid=max(data[2])
        if max_seesionid < data[3]:
            max_seesionid = data[3]
       

    return max_userid, max_itemid, max_seesionid

def load_data(dataset,data_path,neg_len=0):
    
    if neg_len == 0: neg_str=''
    else: neg_str = str(neg_len)

    print('neg length: ', neg_len)

    if not os.path.exists(os.path.join(data_path,dataset)+'/train_seq'+neg_str+'.pkl'): 
            # create the tmp filepath to save data
        print('try to build ',os.path.join(data_path,dataset)+'/train_seq.pkl')
        with open(os.path.join(data_path,dataset)+'/train.pkl','rb') as f:
            train_data=pickle.load(f)

        with open(os.path.join(data_path,dataset)+'/test.pkl','rb') as f:
            test_data=pickle.load(f)
        
        
        with open(os.path.join(data_path,dataset)+'/val.pkl','rb') as f:
            val_data=pickle.load(f)
        
        # with open(os.path.join(data_path,dataset)+'/session2hissession.pkl','rb') as f:
        #     his_session = pickle.load(f)
       
        max_user_train, max_item_train, max_session_train = max_num(train_data)
        max_user_val, max_item_val, max_session_val = max_num(val_data)
        max_user_test, max_item_test, max_session_test = max_num(test_data)

        print('The user id / item id / session id of train data:', max_user_train, max_item_train, max_session_train)
        print('The user id / item id / session id of val data:', max_user_val, max_item_val, max_session_val)
        print('The user id / item id / session id of test data:', max_user_test, max_item_test, max_session_test)
        
        max_userid = max(max_user_train, max_user_val, max_user_test)
        max_itemid = max(max_item_train, max_item_val, max_item_test)
        max_sessionid = max(max_session_train, max_session_val, max_session_test)


       
        train_data =common_seq(train_data, max_itemid)
        test_data =common_seq(test_data, max_itemid)
        val_data =common_seq(val_data, max_itemid)
        item_prob_list = get_item_prob_list(train_data, max_itemid)

        with open(os.path.join(data_path,dataset)+'/test_seq'+neg_str+'.pkl','wb') as f:
            pickle.dump(test_data,f)

        with open(os.path.join(data_path,dataset)+'/train_seq'+neg_str+'.pkl','wb') as f:
            pickle.dump(train_data,f)
        
        with open(os.path.join(data_path,dataset)+'/val_seq'+neg_str+'.pkl','wb') as f:
            pickle.dump(val_data,f)
        
        

        return train_data,val_data, test_data, max_userid, max_itemid, max_sessionid, item_prob_list

    
    with open(os.path.join(data_path,dataset)+'/train_seq'+neg_str+'.pkl','rb') as f:
        train_data=pickle.load(f)
        
    with open(os.path.join(data_path,dataset)+'/test_seq'+neg_str+'.pkl','rb') as f:
        test_data=pickle.load(f)
    
    with open(os.path.join(data_path,dataset)+'/val_seq'+neg_str+'.pkl','rb') as f:
        val_data=pickle.load(f)
    
   
    
    max_user_train, max_item_train, max_session_train = get_max_from_seq(train_data)
    max_user_val, max_item_val, max_session_val  = get_max_from_seq(val_data)
    max_user_test, max_item_test, max_session_test  = get_max_from_seq(test_data)

    print('The max user id / item id / session id of train data:', max_user_train, max_item_train, max_session_train)
    print('The max user id / item id / session id of val data:', max_user_val, max_item_val, max_session_val)
    print('The max user id / item id / session id of test data:', max_user_test, max_item_test, max_session_test)
        
    
    max_userid = max(max_user_train, max_user_val, max_user_test)
    max_itemid = max(max_item_train, max_item_val, max_item_test)
    max_sessionid = max(max_session_train, max_session_val, max_session_test)

    item_prob_list = get_item_prob_list(train_data, max_itemid)

    # max_userid = 145749
    # max_itemid = 39113
    return train_data,val_data, test_data, max_userid, max_itemid, max_sessionid, item_prob_list

