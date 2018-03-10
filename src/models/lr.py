import numpy as np

from models.base_model import BaseModel

from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression

class LogisticReg(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs.pop('n_entity')
        self.n_relation = kwargs.pop('n_relation')
        self.dim = kwargs.pop('dim')

        self.train_path = kwargs.pop('train_path') 
        self.ent_vocab = kwargs.pop('ent_vocab')
        self.rel_vocab = kwargs.pop('rel_vocab')

        self.output = kwargs.pop('output')

        self.wv = KeyedVectors.load_word2vec_format(kwargs.pop('wv_model_path'), binary=False)
        self.empty_wv = np.array([0.0 for i in range(self.dim)])

        self.lr = LogisticRegression()

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
            score_mat[i] = self.predict(self.pick_ent(subs[i]), rels[i], 1)
        return score_mat
    
    def cal_rel_scores(self, subs, objs):
        _batchsize = len(subs)
        
        score_mat = np.empty((_batchsize, self.n_relation))
        for i in range(_batchsize):
            score_mat[i] = self.predict(self.pick_ent(subs[i]), self.pick_ent(objs[i]), 0)
        return score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(objs)

        # NOTE: in this way, we do not consider the direction of an edge.
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = self.predict(self.pick_ent(objs[i]), rels[i], -1)
        return score_mat

    def cal_triplet_scores(self, **kwargs):
        raise NotImplementedError

    # task_type = 1 : (h,r,?)
    # task_type = -1: (?,r,t)
    def predict(self, x, y, task_type):
        X = []
        x_list = x.tolist() 
        if task_type == 1:
            for ent_id in range(self.n_entity):
                X.append(x_list + self.pick_ent(ent_id).tolist())
        elif task_type == -1:
            for ent_id in range(self.n_entity):
                X.append(self.pick_ent(ent_id).tolist() + x_list)
        elif task_type == 0:
            X.append(x_list+y.tolist())
        y_prob = self.lr.predict_proba(X)
        if task_type == 0:
            return np.array(y_prob[0])
        y_idx = -1
        for i in range(len(self.lr.classes_)):
            if self.lr.classes_[i] == y:
                y_idx = i
                break
        res = np.array(y_prob)[:,y_idx]
        print('len:{},dist:{}'.format(len(res), res))
        return res

    # train data construction: one-vs-rest; one-vs-one with negative sampling
    # feature combination: stacking
    # predication cases: (h,t)->r; (h,r)->t; (r,t)->h
    def train(self):
        X = []
        y = []
        with open(self.train_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_id = self.ent_vocab[sub]
                rel_id = self.rel_vocab[rel]
                obj_id = self.ent_vocab[obj]

                # one-vs-rest
                X.append(self.pick_ent(sub_id).tolist() + self.pick_ent(obj_id).tolist())
                y.append(rel_id)

        # using the default configurations to predict relation.
        # LR model can only solve the link prediction.
        self.lr.fit(X,y)

