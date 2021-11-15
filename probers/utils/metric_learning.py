import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import random
from torch.cuda.amp import autocast
from pytorch_metric_learning import miners, losses, distances
LOGGER = logging.getLogger(__name__)


class Sap_Metric_Learning_pairwise(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda, 
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", 
            agg_mode="cls", infoNCE_tau="0.04", use_layer=-1):

        LOGGER.info("Sap_Metric_Learning_pairwise! learning_rate={} weight_decay={} use_cuda={} loss={} infoNCE_tau={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,infoNCE_tau,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning_pairwise, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.use_layer = use_layer
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=infoNCE_tau) # sentence: 0.04, word: 0.2, phrase: 0.04  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        elif self.loss == "contrastive_loss":
            self.loss = losses.ContrastiveLoss()
        elif self.loss == "two_term_loss":
            self.loss = losses.ContrastiveLoss(pos_margin=0.0,neg_margin=-10.0)


        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        

        one_pass = False # for probing dropout
        if one_pass:
            query_toks_combined = {"input_ids": torch.cat([query_toks1["input_ids"], query_toks2["input_ids"]], dim=0), 
                    "attention_mask": torch.cat([query_toks1["attention_mask"], query_toks2["attention_mask"]], dim=0)}
            outputs = self.encoder(**query_toks_combined, return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            if self.agg_mode=="cls":
                query_embed = last_hidden_state[:,0]
            elif self.agg_mode == "mean_pool":
                query_embed = last_hidden_state.mean(1)
                #query_embed = (last_hidden_state * query_toks_combined['attention_mask'].unsqueeze(-1)).sum(1) / query_toks_combined['attention_mask'].sum(-1).unsqueeze(-1)
            elif self.agg_mode == "cls_pooler":
                query_embed = pooler_output

            else:
                raise NotImplementedError()

            # print embeddings, check if they are identical (even when dropout is on)
            #print (query_embed[:len(query_toks1["input_ids"])])
            #print (query_embed[len(query_toks1["input_ids"]):])
            #exit()
        else:
            outputs1 = self.encoder(**query_toks1, return_dict=True, output_hidden_states=True)
            outputs2 = self.encoder(**query_toks2, return_dict=True, output_hidden_states=True)
            
            #last_hidden_state1 = outputs1.last_hidden_state
            #last_hidden_state2 = outputs2.last_hidden_state
            last_hidden_state1 = outputs1.hidden_states[self.use_layer]
            last_hidden_state2 = outputs2.hidden_states[self.use_layer]


            #hidden_states1 = outputs1.hidden_states
            #hidden_states2 = outputs2.hidden_states

            #num_layer = 12 # drop only one layer
            #layermix1 = torch.stack(list(np.random.choice(hidden_states1, num_layer)),2).mean(2)
            #layermix2 = torch.stack(list(np.random.choice(hidden_states2, num_layer)),2).mean(2)

            pooler_output1 = outputs1.pooler_output
            pooler_output2 = outputs2.pooler_output

            if self.agg_mode=="cls":
                query_embed1 = last_hidden_state1[:,0]  # query : [batch_size, hidden]
                query_embed2 = last_hidden_state2[:,0]  # query : [batch_size, hidden]
            elif self.agg_mode == "mean_pool":
                query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
                query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
                #query_embed1 = (last_hidden_state1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
                #query_embed2 = (last_hidden_state2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
            elif self.agg_mode == "cls_pooler":
                query_embed1 = pooler_output1
                query_embed2 = pooler_output2
            elif self.agg_mode == "layer_mix":
                raise NotImplementedError()
                #query_embed1 = layermix1.mean(1)
                #query_embed2 = layermix2.mean(1)
            else:
                raise NotImplementedError()
            query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        
        labels = torch.cat([labels, labels], dim=0)
        
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return self.loss(query_embed, labels) 

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


class Sap_Metric_Learning_dual_encoder(nn.Module):
    def __init__(self, encoder_query, encoder_answer, learning_rate, weight_decay, use_cuda, 
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", 
            agg_mode="cls", infoNCE_tau="0.04", use_layer=-1):

        LOGGER.info("Sap_Metric_Learning_dual_encoder! learning_rate={} weight_decay={} use_cuda={} loss={} infoNCE_tau={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,infoNCE_tau,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning_dual_encoder, self).__init__()
        self.encoder_query = encoder_query
        self.encoder_answer = encoder_answer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.use_layer = use_layer
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=infoNCE_tau) # sentence: 0.04, word: 0.2, phrase: 0.04  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        elif self.loss == "contrastive_loss":
            self.loss = losses.ContrastiveLoss()
        elif self.loss == "two_term_loss":
            self.loss = losses.ContrastiveLoss(pos_margin=0.0,neg_margin=-10.0)


        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks1, query_toks2, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        
        outputs1 = self.encoder_query(**query_toks1, return_dict=True, output_hidden_states=True)
        outputs2 = self.encoder_answer(**query_toks2, return_dict=True, output_hidden_states=True)
        
        #last_hidden_state1 = outputs1.last_hidden_state
        #last_hidden_state2 = outputs2.last_hidden_state
        last_hidden_state1 = outputs1.hidden_states[self.use_layer]
        last_hidden_state2 = outputs2.hidden_states[self.use_layer]

        #hidden_states1 = outputs1.hidden_states
        #hidden_states2 = outputs2.hidden_states

        #num_layer = 12 # drop only one layer
        #layermix1 = torch.stack(list(np.random.choice(hidden_states1, num_layer)),2).mean(2)
        #layermix2 = torch.stack(list(np.random.choice(hidden_states2, num_layer)),2).mean(2)

        pooler_output1 = outputs1.pooler_output
        pooler_output2 = outputs2.pooler_output

        if self.agg_mode=="cls":
            query_embed1 = last_hidden_state1[:,0]  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2[:,0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_pool":
            query_embed1 = last_hidden_state1.mean(1)  # query : [batch_size, hidden]
            query_embed2 = last_hidden_state2.mean(1)  # query : [batch_size, hidden]
            #query_embed1 = (last_hidden_state1 * query_toks1['attention_mask'].unsqueeze(-1)).sum(1) / query_toks1['attention_mask'].sum(-1).unsqueeze(-1)
            #query_embed2 = (last_hidden_state2 * query_toks2['attention_mask'].unsqueeze(-1)).sum(1) / query_toks2['attention_mask'].sum(-1).unsqueeze(-1)
        elif self.agg_mode == "cls_pooler":
            query_embed1 = pooler_output1
            query_embed2 = pooler_output2
        else:
            raise NotImplementedError()
        query_embed = torch.cat([query_embed1, query_embed2], dim=0)
        
        labels = torch.cat([labels, labels], dim=0)
        
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return self.loss(query_embed, labels) 



class Sap_Metric_Learning(nn.Module):
    def __init__(self, encoder, learning_rate, weight_decay, use_cuda,  
            loss, use_miner=True, miner_margin=0.2, type_of_triplets="all", agg_mode="cls", infoNCE_tau="0.04"):

        LOGGER.info("Sap_Metric_Learning! learning_rate={} weight_decay={} use_cuda={} loss={} infoNCE_tau={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
            learning_rate,weight_decay,use_cuda,loss,infoNCE_tau,use_miner,miner_margin,type_of_triplets,agg_mode
        ))
        super(Sap_Metric_Learning, self).__init__()
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.optimizer = optim.AdamW([{'params': self.encoder.parameters()},], 
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=infoNCE_tau) # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        elif self.loss == "contrastive_loss":
            self.loss = losses.ContrastiveLoss()
        elif self.loss == "two_term_loss":
            self.loss = losses.ContrastiveLoss(pos_margin=0.0,neg_margin=10.0)

        print ("miner:", self.miner)
        print ("loss:", self.loss)
    
    @autocast() 
    def forward(self, query_toks, labels):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        
        last_hidden_state = self.encoder(**query_toks, return_dict=True).last_hidden_state
        if self.agg_mode=="cls":
            query_embed = last_hidden_state[:,0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_pool":
            query_embed = last_hidden_state.mean(1)  # query : [batch_size, hidden]
        else:
            raise NotImplementedError()
        
        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels)
            return self.loss(query_embed, labels, hard_pairs) 
        else:
            return self.loss(query_embed, labels) 

    def get_embeddings(self, mentions, batch_size=1024):
        """
        Compute all embeddings from mention tokens.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(mentions), batch_size)):
                end = min(start + batch_size, len(mentions))
                batch = mentions[start:end]
                batch_embedding = self.vectorizer(batch)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table
