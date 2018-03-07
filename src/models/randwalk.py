import math
import numpy as np

from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *

from models.deepwalk_rdf.deepwalk import deepwalk
from gensim.models import KeyedVectors

class RandWalk(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs.pop('n_entity')
        self.n_relation = kwargs.pop('n_relation')
        self.dim = kwargs.pop('dim')
        self.knowledge_path = kwargs.pop('knowledge_path')
        self.ent_vocab = kwargs.pop('ent_vocab')
        self.rel_vocab = kwargs.pop('rel_vocab')
        self.output = kwargs.pop('output')
        self.empty_wv = np.array([0.0 for i in range(self.dim)])

    def cal_rank(self, **kwargs):
        raise NotImplementedError

    # For max-margin loss
    def _pairwisegrads(self, **kwargs):
        raise NotImplementedError

    # For log-likelihood
    def _singlegrads(self, **kwargs):
        raise NotImplementedError

    def _composite(self, **kwargs):
        raise NotImplementedError

    def _cal_similarity(self, **kwargs):
        raise NotImplementedError

    def pick_ent(self, ents):
        if str(ents) in self.wv:
            return self.wv[str(ents)]
        else:
            return np.array(self.empty_wv)

    def pick_rel(self, rels):
        if str(rels) in self.wv:
            return self.wv[str(rels)]
        else:
            return np.array(self.empty_wv) # NOTE: delete the np.array later!!

    def cal_scores(self, subs, rels):
        _batchsize = len(subs)
        
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = self.predict(self.pick_ent(subs[i]), self.pick_rel(rels[i]+self.n_entity))
        return score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(objs)

        # NOTE: in this way, we do not consider the direction of an edge.
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = self.predict(self.pick_ent(objs[i]), self.pick_rel(rels[i]+self.n_entity))
        return score_mat

    def cal_triplet_scores(self, **kwargs):
        raise NotImplementedError

    # inefficent now. use matrix operation later.
    def predict(self, ent_emb, rel_emb):
        score = np.zeros(self.n_entity)
        norm_ent = 0.0
        norm_rel = 0.0
        for i in range(self.n_entity):
            target_emb = self.pick_ent(i)
            e2t = np.dot(ent_emb, target_emb)
            r2t = np.dot(rel_emb, target_emb)
            score[i] = e2t + r2t
            norm_ent = norm_ent + np.exp(e2t)
            norm_rel = norm_rel + np.exp(r2t)
        score = score - np.log(norm_ent)
        score = score - np.log(norm_rel)
        return score
    
    def load_wv_model(self, wv_model_path):
        self.wv = KeyedVectors.load_word2vec_format(wv_model_path, binary=False)
        return

    def train(self):
        edgelist = []
        with open(self.knowledge_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                edgelist.append((self.ent_vocab[sub], self.ent_vocab[obj], int(self.rel_vocab[rel])+self.n_entity))
        deepwalk.run(edgelist=edgelist,
                format="memory",
                dim=self.dim,
                output=self.output)

