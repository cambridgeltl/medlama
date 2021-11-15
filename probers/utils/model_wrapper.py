import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from .metric_learning import *

from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

LOGGER = logging.getLogger()


class Model_Wrapper(object):
    """
    Wrapper class for BERT encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)
        

        # save bert vocab
        self.tokenizer.save_pretrained(path)
        

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder = self.encoder.cuda()

        return self.encoder, self.tokenizer
    

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min() ) / (score_matrix.max() - score_matrix.min())
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i+batch_size]).cuda()
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None: 
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], axis=0)

        return res.numpy()

    def embed_dense(self, names, show_progress=False, batch_size=2048, agg_mode="cls"):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=batch_size
        dense_embeds = []

        #print ("converting names to list...")
        #names = names.tolist()

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)
                
            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                        batch, add_special_tokens=True, 
                        truncation=True, max_length=25, 
                        padding="max_length", return_tensors='pt')
                batch_tokenized_names_cuda = {}
                for k,v in batch_tokenized_names.items(): 
                    batch_tokenized_names_cuda[k] = v.cuda()
                
                if agg_mode == "cls":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0][:,0,:] # [CLS]
                elif agg_mode == "mean_pool":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0].mean(1) # pooling
                else:
                    print ("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds

class Model_Wrapper_dual_encoder(object):
    """
    Wrapper class for dual-BERT encoder
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder_query = None
        self.encoder_answer = None

    def save_model(self, path):
        # save bert model, bert config
        os.mkdir(os.path.join(path, "query"))
        os.mkdir(os.path.join(path, "answer"))
        self.encoder_query.save_pretrained(os.path.join(path, "query"))
        self.encoder_answer.save_pretrained(os.path.join(path, "answer"))
        
        # save bert vocab
        self.tokenizer.save_pretrained(os.path.join(path, "query"))
        self.tokenizer.save_pretrained(os.path.join(path, "answer"))
        

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase)
        self.encoder_query = AutoModel.from_pretrained(path)
        self.encoder_answer = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder_query = self.encoder_query.cuda()
            self.encoder_answer = self.encoder_answer.cuda()

        return self.encoder_query, self.encoder_answer, self.tokenizer
    

    def get_score_matrix(self, query_embeds, dict_embeds, cosine=False, normalise=False):
        """
        Return score matrix

        Parameters
        ----------
        query_embeds : np.array
            2d numpy array of query embeddings
        dict_embeds : np.array
            2d numpy array of query embeddings

        Returns
        -------
        score_matrix : np.array
            2d numpy array of scores
        """
        if cosine:
            score_matrix = cosine_similarity(query_embeds, dict_embeds)
        else:
            score_matrix = np.matmul(query_embeds, dict_embeds.T)

        if normalise:
            score_matrix = (score_matrix - score_matrix.min() ) / (score_matrix.max() - score_matrix.min())
        
        return score_matrix

    def retrieve_candidate(self, score_matrix, topk):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """
        
        def indexing_2d(arr, cols):
            rows = np.repeat(np.arange(0,cols.shape[0])[:, np.newaxis],cols.shape[1],axis=1)
            return arr[rows, cols]

        # get topk indexes without sorting
        topk_idxs = np.argpartition(score_matrix,-topk)[:, -topk:]

        # get topk indexes with sorting
        topk_score_matrix = indexing_2d(score_matrix, topk_idxs)
        topk_argidxs = np.argsort(-topk_score_matrix) 
        topk_idxs = indexing_2d(topk_idxs, topk_argidxs)

        return topk_idxs

    def retrieve_candidate_cuda(self, score_matrix, topk, batch_size=128, show_progress=False):
        """
        Return sorted topk idxes (descending order)

        Parameters
        ----------
        score_matrix : np.array
            2d numpy array of scores
        topk : int
            The number of candidates

        Returns
        -------
        topk_idxs : np.array
            2d numpy array of scores [# of query , # of dict]
        """

        res = None
        for i in tqdm(np.arange(0, score_matrix.shape[0], batch_size), disable=not show_progress):
            score_matrix_tmp = torch.tensor(score_matrix[i:i+batch_size]).cuda()
            matrix_sorted = torch.argsort(score_matrix_tmp, dim=1, descending=True)[:, :topk].cpu()
            if res is None: 
                res = matrix_sorted
            else:
                res = torch.cat([res, matrix_sorted], axis=0)

        return res.numpy()

    def embed_dense(self, names, show_progress=False, batch_size=2048, agg_mode="cls"):
        """
        Embedding data into dense representations

        Parameters
        ----------
        names : np.array
            An array of names

        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """
        self.encoder.eval() # prevent dropout
        
        batch_size=batch_size
        dense_embeds = []

        #print ("converting names to list...")
        #names = names.tolist()

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, len(names), batch_size))
            else:
                iterations = range(0, len(names), batch_size)
                
            for start in iterations:
                end = min(start + batch_size, len(names))
                batch = names[start:end]
                batch_tokenized_names = self.tokenizer.batch_encode_plus(
                        batch, add_special_tokens=True, 
                        truncation=True, max_length=25, 
                        padding="max_length", return_tensors='pt')
                batch_tokenized_names_cuda = {}
                for k,v in batch_tokenized_names.items(): 
                    batch_tokenized_names_cuda[k] = v.cuda()
                
                if agg_mode == "cls":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0][:,0,:] # [CLS]
                elif agg_mode == "mean_pool":
                    batch_dense_embeds = self.encoder(**batch_tokenized_names_cuda)[0].mean(1) # pooling
                else:
                    print ("no such agg_mode:", agg_mode)

                batch_dense_embeds = batch_dense_embeds.cpu().detach().numpy()
                dense_embeds.append(batch_dense_embeds)
        dense_embeds = np.concatenate(dense_embeds, axis=0)
        
        return dense_embeds
