import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv
# from gnn import GCNConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter_mean
import dgl.nn.pytorch as dglnn
from module import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_





class Session_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(Session_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank):
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank)
        att_mask = torch.where(att_mask, 0., -1e9)
        return self.transformer_encoder(input_embs, log_mask, att_mask)


    
class ID_module(nn.Module):
   
    def __init__(self, item_num, id_in_channels, out_channels,args ):
        super(ID_module, self).__init__()
        self.emb = torch.nn.Embedding(item_num, id_in_channels)
        self.layernorm1 = torch.nn.LayerNorm(out_channels)
        self.session_encoder_id = Session_Encoder(
            item_num=item_num,
            max_seq_len=args.max_len,
            item_dim=out_channels,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout,
            n_layers=args.transformer_block)
        self.args = args
    
    def get_sess(self, item_emb_feat, mask, device):

        if self.args.id_module == 'average':
            seq_emb = torch.sum(mask.unsqueeze(2)*item_emb_feat, 1)
            seq_emb = seq_emb/(mask.sum(dim=1)).unsqueeze(1)
            session_emb_feat = seq_emb 

        elif self.args.id_module == 'transformer':
            trans_emb =  self.session_encoder_id(item_emb_feat, mask, device)
            index =  mask.sum(1)-1
            session_emb_feat = torch.gather(trans_emb, 1, index.unsqueeze(1).unsqueeze(2).expand(-1,-1,trans_emb.size(2))).squeeze(1)
            

        return session_emb_feat
    
    def forward(self, itemid, mask):
        emb = self.emb.weight

        emb = self.layernorm1(emb)

        device = itemid.device


        item_emb = emb[itemid] 

        ###
        if self.args.id_module == 'average':
            seq_emb = torch.sum(mask.unsqueeze(2)*item_emb, 1)
            seq_emb = seq_emb/(mask.sum(dim=1)).unsqueeze(1)
            session_emb = seq_emb 

        ##
        elif self.args.id_module == 'transformer':
            trans_emb =  self.session_encoder_id(item_emb, mask, device)
            index =  mask.sum(1)-1
            session_emb = torch.gather(trans_emb, 1, index.unsqueeze(1).unsqueeze(2).expand(-1,-1,trans_emb.size(2))).squeeze(1)
    
        return session_emb, emb

    
        
class Text_module(nn.Module):
   
    def __init__(self, item_num, lm_in_channels, out_channels, args, dropout):
        super(Text_module, self).__init__()

        self.linearlayer1 = torch.nn.Linear(lm_in_channels, out_channels)
        self.layernorm2 = torch.nn.LayerNorm(out_channels)
        self.dropout = dropout

        self.session_encoder = Session_Encoder(
            item_num=item_num,
            max_seq_len=args.max_len,
            item_dim=out_channels,
            num_attention_heads=args.num_attention_heads,
            dropout=args.dropout,
            n_layers=args.transformer_block)
        
    def map_feat(self, feat, func):
    
        new_item = func(feat)
        new_item = F.relu(new_item)
        new_item = F.dropout(new_item, p=self.dropout, training=self.training)

        return new_item

    def get_sess(self, item_emb_feat, mask, device):
        trans_emb =  self.session_encoder(item_emb_feat, mask, device)
        index =  mask.sum(1)-1
        session_emb_feat = torch.gather(trans_emb, 1, index.unsqueeze(1).unsqueeze(2).expand(-1,-1,trans_emb.size(2))).squeeze(1)
            
        return session_emb_feat


    def forward(self, x, itemid, mask, device):

        xmap =  self.map_feat(x, self.linearlayer1)
       
        xmap = self.layernorm2(xmap)

        item_emb_feat = xmap[itemid]

        trans_emb =  self.session_encoder(item_emb_feat, mask, device)
        index =  mask.sum(1)-1
        session_emb_feat = torch.gather(trans_emb, 1, index.unsqueeze(1).unsqueeze(2).expand(-1,-1,trans_emb.size(2))).squeeze(1)

        return session_emb_feat, xmap



    
class AlterRec(nn.Module):
   
    def __init__(self, node_num, item_num, id_in_channels, lm_in_channels, out_channels, num_layers,
                 dropout, args,  feat=None, item_prob_list=None):
        super(AlterRec, self).__init__()

        self.out_channels = out_channels
        
        self.dropout = dropout
   
        self.args= args
        self.feat = feat
        self.item_prob_list = item_prob_list

        
        self.id_module = ID_module(item_num, id_in_channels, out_channels,args )
        self.text_module = Text_module(item_num, lm_in_channels, out_channels, args, dropout)
        self.linearlayer = torch.nn.Linear(2, 1)


    def reset_parameters(self):
       
    
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
  
    
  
    def get_text_score(self, xmap,itemid, mask, device, max_itemid, mode ):
        item_emb_feat = xmap[itemid]

        if mode == 'text':
            session_emb_feat =  self.text_module.get_sess(item_emb_feat, mask, device)
            
        elif  mode == 'gnn':
            session_emb_feat =  self.id_module.get_sess(item_emb_feat, mask, device)
           
            
        item_embs_feat = xmap[:max_itemid+1]
        scores = torch.matmul(session_emb_feat, item_embs_feat.permute(1, 0))  

        return scores



    
    def forward(self, x,  max_itemid, itemid, mask, label_hard_neg_id=None, mask_hard_id=None, label_hard_neg_text=None, mask_hard_text=None, train_gnn=None, train_lm=None):
        
       
        device = itemid.device

        ### ID-based
        session_emb, gnn_emb = self.id_module(itemid, mask)
       

        item_embs = gnn_emb[:max_itemid+1]
        scores1 =  torch.matmul(session_emb, item_embs.permute(1, 0))    


        if self.training and not self.args.full_neg and train_gnn == 1:
            x_ind = torch.arange(scores1.size(0)).unsqueeze(1).repeat(1, label_hard_neg_id.size(1)).to(device)
            scores1 = scores1[x_ind, label_hard_neg_id]
            
            scores1[mask_hard_id] =  -1e5



        ### text-based
        session_emb_feat, xmap = self.text_module( x, itemid, mask, device)
        item_embs_feat = xmap[:max_itemid+1]
        scores2 = torch.matmul(session_emb_feat, item_embs_feat.permute(1, 0))  /  self.args.temperature   



        if self.training and not self.args.full_neg and train_lm == 1:
            x_ind = torch.arange(scores2.size(0)).unsqueeze(1).repeat(1, label_hard_neg_text.size(1)).to(device)
            scores2 = scores2[x_ind, label_hard_neg_text]
            scores2[mask_hard_text] =  -1e5
       

        if self.training:
            scores = torch.tensor(0).to(device)
        else:
            scores = self.args.alpha*(scores1) + (1-self.args.alpha)*(scores2)

       

        score_gnn = scores1
        score_text = scores2

        if self.args.only_train_id:
            scores = score_gnn
        
        elif self.args.only_train_text:
            scores = score_text

       
      
        return scores, score_gnn, score_text, item_embs, gnn_emb[:max_itemid+1], xmap[:max_itemid+1]
    

