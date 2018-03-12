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

        # external learned embedding, e.g., deepwalk
        self.wv = KeyedVectors.load_word2vec_format(kwargs.pop('wv_model_path'), binary=False)
        self.empty_wv = np.array([0.0 for i in range(self.dim)])

        self.lr = LogisticRegression(verbose=1)
        self.cls_type = ""
        self.feat_type = kwargs.pop('feat_type')
        self.negative = kwargs.pop('negative')

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
            score_mat[i] = self.predict(subs[i], rels[i], None)
        return score_mat
    
    def cal_rel_scores(self, subs, objs):
        _batchsize = len(subs)
        
        score_mat = np.empty((_batchsize, self.n_relation))
        for i in range(_batchsize):
            score_mat[i] = self.predict(subs[i], None, objs[i])
        return score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(objs)

        # NOTE: in this way, we do not consider the direction of an edge.
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = self.predict(None, rels[i], objs[i])
        return score_mat

    def cal_triplet_scores(self, **kwargs):
        raise NotImplementedError

    # computing feature
    # task_type = 1: link prediction
    # task_type = 0: triple classification
    def compute_feature(self, subs, rels, objs, task_type):
        if task_type == 1:
            if self.feat_type == "concate":
                return subs.tolist() + objs.tolist()
            else: # translation model
                return (subs-objs).tolist()
        elif task_type == 0:
            if self.feat_type == "concate":
                return subs.tolist() + rels.tolist() + objs.tolist()
            else: # translation model
                return (subs + rels - objs).tolist()

    # task_type = 1 : (h,r,?)
    # task_type = -1: (?,r,t)
    def predict(self, subs, rels, objs):
        if self.cls_type == "triplet_clssifer":
            return self.binary_predictor(subs, rels, objs)
        else:
            return self.multi_predictor(subs, rels, objs)

    def multi_predictor(self, subs, rels, objs):
        X = []
        if objs == None:
            sub_emb = self.pick_ent(subs)
            for ent_id in range(self.n_entity):
                obj_emb = self.pick_ent(ent_id)
                X.append(self.compute_feature(sub_emb, None, obj_emb, 1))
        elif subs == None:
            obj_emb = self.pick_ent(objs)
            for ent_id in range(self.n_entity):
                sub_emb = self.pick_ent(ent_id)
                X.append(self.compute_feature(sub_emb, None, obj_emb, 1))
        elif rels == None:
            sub_emb = self.pick_ent(subs)
            obj_emb = self.pick_ent(objs)
            X.append(self.compute_feature(sub_emb, None, obj_emb, 1))

        y_prob = self.lr.predict_proba(X)
        if rels == None:
            return np.array(y_prob[0])
        y_idx = -1
        for i in range(len(self.lr.classes_)):
            if self.lr.classes_[i] == rels:
                y_idx = i
                break
        res = np.array(y_prob)[:,y_idx]
        # print('len:{},dist:{}'.format(len(res), res))
        return res

    def binary_predictor(self, subs, rels, objs):
        X = []
        if objs == None:
            sub_emb = self.pick_ent(subs)
            rel_emb = self.pick_rel(rels)
            for ent_id in range(self.n_entity):
                obj_emb = self.pick_ent(ent_id)
                X.append(self.compute_feature(sub_emb, rel_emb, obj_emb, 0))
        elif subs == None:
            obj_emb = self.pick_ent(objs)
            rel_emb = self.pick_rel(rels)
            for ent_id in range(self.n_entity):
                sub_emb = self.pick_ent(ent_id)
                X.append(self.compute_feature(sub_emb, rel_emb, obj_emb, 0))
        elif rels == None:
            sub_emb = self.pick_ent(subs)
            obj_emb = self.pick_rel(objs)
            for rel_id in range(self.n_relation):
                rel_emb = self.pick_rel(rel_id)
                X.append(self.compute_feature(sub_emb, rel_emb, obj_emb, 0))
        y_prob = self.lr.predict_proba(X)
        y_idx = -1
        for i in range(len(self.lr.classes_)):
            if self.lr.classes_[i] == 1: # 1 means postive class
                y_idx = i
                break
        res = np.array(y_prob)[:, self.lr.classes_[y_idx]] 
        # print('len:{}, dist:{}'.format(len(res), res))
        return res

    def check_triple(self, subs, rels, objs):
        s = len(subs)
        X = []
        for idx in range(s):
            X.append(self.pick_ent(subs[idx]).tolist() + self.pick_rel(rels[idx]+self.n_entity).tolist() + self.pick_ent(objs[idx]).tolist())
        y = np.array(self.lr.predict(X))
        res = np.sum(y == 1)
        return res

    # train data construction: one-vs-rest; one-vs-one with negative sampling
    # feature combination: stacking
    # predication cases: (h,t)->r; (h,r)->t; (r,t)->h
    def train(self):
        self.cls_type = "link_predict"
        X = []
        y = []
        with open(self.train_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_id = self.ent_vocab[sub]
                rel_id = self.rel_vocab[rel]
                obj_id = self.ent_vocab[obj]

                # one-vs-rest
                sub_emb = self.pick_ent(sub_id)
                obj_emb = self.pick_ent(obj_id)
                X.append(self.compute_feature(sub_emb, None, obj_emb, 1))
                y.append(rel_id)

        # using the default configurations to predict relation.
        # LR model can only solve the link prediction.
        self.lr.fit(X,y)

    def train_triple_classifer(self):
        self.cls_type = "triplet_clssifer"
        X = []
        y = []
        generator = UniformNegativeGenerator(self.n_entity, self.negative) 
        pos_triplets = [] 
        with open(self.train_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_id = self.ent_vocab[sub]
                rel_id = self.rel_vocab[rel]
                obj_id = self.ent_vocab[obj]
                pos_triplets.append((sub_id, rel_id, obj_id))

                # postive example
                sub_emb = self.pick_ent(sub_id)
                rel_emb = self.pick_rel(rel_id)
                obj_emb = self.pick_ent(obj_id)
                X.append(self.compute_feature(sub_emb, rel_emb, obj_emb, 0))
                y.append(1)

        # negative example
        neg_triplets = generator.generate(np.array(pos_triplets))
        for neg in neg_triplets:
            sub_id = neg[0]
            rel_id = neg[1]
            obj_id = neg[2]
            
            sub_emb = self.pick_ent(sub_id)
            rel_emb = self.pick_rel(rel_id)
            obj_emb = self.pick_ent(obj_id)
            X.append(self.compute_feature(sub_emb, rel_emb, obj_emb, 0))
            y.append(-1)

        # using the default configurations to predict relation.
        # LR model can only solve the link prediction.
        self.lr.fit(X,y)

# sampling negative samples
class NegativeGenerator(object):
    def __init__(self, n_ent, n_negative, train_graph=None):
        self.n_ent = n_ent
        self.n_negative = n_negative
        if train_graph:
            raise NotImplementedError
        self.graph = train_graph  # for preventing from including positive triplets as negative ones

    def generate(self, pos_triplets):
        """
        :return: neg_triplets, whose size is (length of positives \times n_sample , 3)
        """
        raise NotImplementedError

class UniformNegativeGenerator(NegativeGenerator):
    def __init__(self, n_ent, n_negative, train_graph=None):
        super(UniformNegativeGenerator, self).__init__(n_ent, n_negative, train_graph)

    def generate(self, pos_triplets):
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * self.n_negative
        neg_ents = np.random.randint(0, self.n_ent, size=sample_size)
        neg_triplets = np.tile(pos_triplets, (self.n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets