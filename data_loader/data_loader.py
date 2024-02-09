
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
               

        # count += 1

    
    final_seqs=[]
    for i in range(len(uid)):
        sess = sessionid[i]
        userid = uid[i]
        # if userid in his_session:
        #     s2h = his_session[userid]
        #     if sess in s2h:
        #         his_sess = s2h[sess]
        #     else:
        #         his_sess = [sess]
        
      
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

def sample_relations(dataset_name, num, data_path, sample_size=20):
    """
    ## edge of item-item if items in the same session
    """
    num  = num + 1
    adj1 = [dict() for _ in range(num)]

    adj = [[] for _ in range(num)]
    
   
    relation = []
    print('item num: ', num)
    with open(os.path.join(data_path, dataset_name)+'/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    for u in tqdm(graph, desc='build the graph...', leave=False):
        u_seqs = graph[u]
        for s, item in u_seqs.items():
            for i in range(len(item) - 1):
                relation.append([item[i], item[i + 1]])
                relation.append([item[i + 1], item[i]])
  
   
    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            # print(tup[0], tup[1])
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1


    for t in range(1, num):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]

    # edge sampling 
    for i in range(1, num):
        adj[i] = adj[i][:sample_size]
  
    # if os.path.exists(f'./dataset/{dataset_name}/adj_{sample_size}.pkl'):
    if os.path.exists(os.path.join(data_path, dataset_name)+f'/adj_{sample_size}.pkl'):
        return 
    
    with open(os.path.join(data_path, dataset_name)+f'/adj_{sample_size}.pkl', 'wb') as f:
        pickle.dump(adj, f)

def construct_graph(dataset_name, sample_size, data_path, max_item_id):

    sample_relations(dataset_name, max_item_id, data_path, sample_size)

    pre = []
    nxt = []
    src_v = []
    dst_u = []
    # build i2i / u2u relations
    
    with open(os.path.join(data_path, dataset_name)+'/train.pkl', 'rb') as f:
        graph = pickle.load(f)

    with open(os.path.join(data_path, dataset_name)+f'/adj_{sample_size}.pkl', 'rb') as f:
        adj = pickle.load(f)
    
    ## sample graph
    for i in range(len(adj)):
      
        _pre = []
        _nxt = []
        for item in adj[i]:   ## edges of item-item in different session 
            _pre.append(i)
            _nxt.append(item)
        pre += _pre
        nxt += _nxt
   

    for u in tqdm(graph, desc='build the graph...', leave=False):  ### edges 
        u_seqs = graph[u]
        for sid, s in u_seqs.items():
            pre += s[:-1]   ## edges of item-item in the same session 
            nxt += s[1:]
            dst_u += [u for _ in s]     ### edges of user-item
            src_v += s

    u_src = []
    u_dst = []
   
   
    item_num = max(max(pre), max(nxt)) +1
    print('addiotn item num', item_num)
   
    
   
    dst_u = [u + item_num for u in dst_u]

    G = dgl.graph((pre, nxt))   ##  edges of item-item in the same session
    G = dgl.add_edges(G, nxt, pre) ## edges of item-item in the same session
    G = dgl.add_edges(G, dst_u, src_v)  ### edges of user-item
    G = dgl.add_edges(G, src_v, dst_u)
    
   

    cat_edge1 = pre + nxt
    cat_edge2 = nxt + pre
    cat_edge3 = dst_u + src_v   ### no user
    cat_edge4 = src_v + dst_u

    ###
    # tmp_edge1 = coalesce(torch.cat([torch.tensor(pre).unsqueeze(0), torch.tensor(nxt).unsqueeze(0)],dim=0 ))
    # tmp_edge2 = coalesce(torch.cat([torch.tensor(dst_u).unsqueeze(0), torch.tensor(src_v).unsqueeze(0)],dim=0 ))
    # tmp_edge3 =  coalesce(torch.cat([tmp_edge1, tmp_edge2],dim=1 ))
    # tmp_edge4 = torch.cat([tmp_edge3, tmp_edge3[torch.tensor([1,0])]],dim=1 )
    # tmp_edge5 = coalesce(tmp_edge4)

    edges1 = torch.cat([torch.tensor(cat_edge1).unsqueeze(0), torch.tensor(cat_edge2).unsqueeze(0)], dim=0)
    edges2 = torch.cat([torch.tensor(cat_edge4).unsqueeze(0), torch.tensor(cat_edge3).unsqueeze(0)], dim=0)

    edges = torch.cat([edges1, edges2], dim=1)

    edge_index = coalesce(edges)
    

    G=dgl.add_self_loop(G)

    print('number of training edges: ', edge_index.size(1))

    return edge_index, G


# def construct_graph(dataset,data_path, max_item_id, SZ):

#     sample_relations(dataset, max_item_id, data_path, sample_size=20)

#     with open(os.path.join(data_path,dataset)+'/train.pkl','rb') as f:
#             train_data=pickle.load(f)
#     with open(os.path.join(data_path,dataset)+'/val.pkl','rb') as f:
#             val_data=pickle.load(f)
#     with open(os.path.join(data_path,dataset)+'/test.pkl','rb') as f:
#             test_data=pickle.load(f)

#     # with open(os.path.join(data_path,dataset)+'/session2hissession.pkl','rb') as f:
#     #         his_session = pickle.load(f)

#     train_edge = []
#     for u in tqdm(train_data,desc='gen_session_item_edge...',leave=False):
#         u_seqs=train_data[u]
#         for sess, seq in u_seqs.items():      
#             for i in range(len(seq)-1):
#                 ### item-item
#                 train_edge.append([seq[i], seq[i + 1]])
#                 train_edge.append([seq[i + 1], seq[i]])

#                 ### user-item
#                 train_edge.append([u + max_item_id, seq[i]])
#                 train_edge.append([seq[i], u + max_item_id])
   
#             train_edge.append([u + max_item_id, seq[i + 1]])
#             train_edge.append([seq[i + 1], u + max_item_id])


#     return train_edge

class SessionDataset(Dataset):
    def __init__(self, data,max_item_id, max_len,  max_his):
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        data format:
        <[uid]> <[v1,v2,v3]> <label> <session_id> <[his_s1, his_s2, his_s3]> <auginstance>
        """
     
        
        data=self.data[index]
        uid=torch.tensor([data[0]]) + self.max_item_id+1
        # sessid = data[3]
        # sessid = sessid + self.max_item_id+1
        # sessid=torch.tensor([sessid])
        
       
        label=torch.tensor(data[2])
        


        browsed_ids=np.zeros((self.max_seq_len),dtype=np.int64)
        seq_len=len(data[1][-self.max_seq_len:])
        browsed_ids[:seq_len]=np.array(data[1][-self.max_seq_len:])
        browsed_ids = torch.tensor(browsed_ids)
        pos_idx=torch.tensor(np.array([seq_len-i-1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int64))

        seq_len=torch.tensor(np.array(seq_len,dtype=np.int64))
        mask=torch.tensor(np.array([1 for i in range(seq_len)]+[ 0 for i in range(self.max_seq_len-seq_len)],dtype=np.int64))
        
        # auginstance = data[5]
        # if auginstance == -1: use_gnn = 0
        # else: use_gnn = 1
        # use_gnn = torch.tensor(use_gnn)
        
        

        return uid, browsed_ids, label, mask
    
    @staticmethod
    def collate_fn(data):
        uid = torch.cat([_[0] for _ in data], dim=0)
        browsed_ids = torch.cat([_[1].unsqueeze(0) for _ in data], dim=0)
        label = torch.cat([_[2] for _ in data], dim=0)
        # session_edge = torch.cat([_[3] for _ in data], dim=0)
        # no_his = torch.cat([_[4] for _ in data], dim=0)
        mask = torch.cat([_[3].unsqueeze(0) for _ in data], dim=0)
        # use_gnn = torch.cat([_[6].unsqueeze(0) for _ in data], dim=0)
        # his_sess = torch.cat([_[7].unsqueeze(0) for _ in data], dim=0)
        # his_mask = torch.cat([_[8].unsqueeze(0) for _ in data], dim=0)
        # pos_idx = torch.cat([_[9].unsqueeze(0) for _ in data], dim=0)


        return uid, browsed_ids, label,  mask





       
         



       
         