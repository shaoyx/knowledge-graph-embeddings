
from models.base_model import BaseModel
from models.param import LookupParameter
from utils.math_utils import *
import queue


class TransE(BaseModel):
    def __init__(self, **kwargs):
        self.n_entity = kwargs.pop('n_entity')
        self.n_relation = kwargs.pop('n_relation')
        self.dim = kwargs.pop('dim')
        self.margin = kwargs.pop('margin')
        mode = kwargs.pop('mode', 'pairwise')
        if mode == 'pairwise':
            self.compute_gradients = self._pairwisegrads
        elif mode == 'single':
            self.compute_gradients = self._singlegrads
        else:
            raise NotImplementedError

        self.params = {'e': LookupParameter(name='e', shape=(self.n_entity, self.dim)),
                       'r': LookupParameter(name='r', shape=(self.n_relation, self.dim))}

    def _pairwisegrads(self, pos_samples, neg_samples):
        assert pos_samples.shape == neg_samples.shape
        self.prepare()
        p_scores = self.cal_triplet_scores(pos_samples)
        n_scores = self.cal_triplet_scores(neg_samples)

        loss = max_margin(p_scores, n_scores)
        idxs = np.where(loss > 0)[0]
        if len(idxs) != 0:
            # TODO: inefficient calculation
            pos_subs, pos_rels, pos_objs = pos_samples[idxs, 0], pos_samples[idxs, 1], pos_samples[idxs, 2]
            neg_subs, neg_rels, neg_objs = neg_samples[idxs, 0], neg_samples[idxs, 1], neg_samples[idxs, 2]

            p_s_embs = self.pick_ent(pos_subs)
            p_r_embs = self.pick_rel(pos_rels)
            p_o_embs = self.pick_ent(pos_objs)
            n_s_embs = self.pick_ent(neg_subs)
            n_r_embs = self.pick_rel(neg_rels)
            n_o_embs = self.pick_ent(neg_objs)

            p_qs = self._composite(p_s_embs, p_r_embs)
            n_qs = self._composite(n_s_embs, n_r_embs)

            p_s_grads = 2 * (p_qs - p_o_embs)
            p_r_grads = 2 * (p_qs - p_o_embs)
            p_o_grads = -2 * (p_qs - p_o_embs)

            n_s_grads = -2 * (n_qs - n_o_embs)
            n_r_grads = -2 * (n_qs - n_o_embs)
            n_o_grads = 2 * (n_qs - n_o_embs)

            _batchsize = len(pos_subs)

            for idx in range(_batchsize):
                self.params['e'].add_grad(pos_subs[idx], p_s_grads[idx])
                self.params['r'].add_grad(pos_rels[idx], p_r_grads[idx])
                self.params['e'].add_grad(pos_objs[idx], p_o_grads[idx])
                self.params['e'].add_grad(neg_subs[idx], n_s_grads[idx])
                self.params['r'].add_grad(neg_rels[idx], n_r_grads[idx])
                self.params['e'].add_grad(neg_objs[idx], n_o_grads[idx])

        else:
            pass

        self.params['e'].finalize()
        self.params['r'].finalize()

        return loss.mean()

    def _singlegrads(self, samples, ys):
        raise NotImplementedError('Only pairwise setting is available')

    def _composite(self, sub_emb, rel_emb):
        return sub_emb + rel_emb

    def _cal_similarity(self, query, obj_emb):
        return - np.sum((query - obj_emb)**2, axis=1)

    def cal_scores(self, subs, rels):
        _batchsize = len(subs)
        sub_emb = self.pick_ent(subs)
        rel_emb = self.pick_rel(rels)
        qs = self._composite(sub_emb, rel_emb)

        # TODO: maybe inefficient. use matrix operation
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = - np.linalg.norm(qs[i] - self.pick_ent(np.arange(self.n_entity)), axis=1) ** 2
        return score_mat

    def cal_scores_inv(self, rels, objs):
        _batchsize = len(objs)
        obj_emb = self.pick_ent(objs)
        rel_emb = self.pick_rel(rels)
        qs_inv = rel_emb - obj_emb

        # TODO: maybe inefficient. use matrix operation
        score_mat = np.empty((_batchsize, self.n_entity))
        for i in range(_batchsize):
            score_mat[i] = - np.linalg.norm(self.pick_ent(np.arange(self.n_entity)) + qs_inv[i], axis=1) ** 2
        return score_mat

    def cal_triplet_scores(self, samples):
        subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
        sub_emb = self.pick_ent(subs)
        rel_emb = self.pick_rel(rels)
        obj_emb = self.pick_ent(objs)
        qs = self._composite(sub_emb, rel_emb)
        return self._cal_similarity(qs, obj_emb)

    def pick_ent(self, ents):
        return self.params['e'].data[ents]

    def pick_rel(self, rels):
        return self.params['r'].data[rels]

    def analyze(self, kb, subs, rels, objs):
        for idx in range(len(subs)):
            sub = subs[idx]
            rel = rels[idx]
            obj = objs[idx]

            sub_emb = self.pick_ent(sub)
            rel_emb = self.pick_rel(rel)
            obj_emb = self.pick_ent(obj)

            # compute rank and top-1 relations
            pred_rel_emb = obj_emb - sub_emb
            sim = np.sum((pred_rel_emb - rel_emb)**2)
            best_rel, best_rel_score, rank = self.search_rel_by_emb(pred_rel_emb, sim)
            print("true_rel:{0}, true2pred_sim: {1:.6f}, rank: {2}, best_rel: {3}, best_rel_score: {4:.6f}".format(rel, sim, rank, best_rel, best_rel_score))

            # do graph search
            vis = [False for i in range(kb.size)]
            vis[sub] = True
            self.dfs_search([(sub, 'STR')], [0. for i in range(self.dim)], vis, kb, obj, sub_emb, rel_emb, obj_emb, 10)

    def search_rel_by_emb(self, pred_rel_emb, rel_score):
        rel_embs = self.params['r'].data
        best_rel = -1
        best_rel_score = np.inf
        rank = 1
        for rel in range(len(rel_embs)):
            score = np.sum((rel_embs[rel] - pred_rel_emb)**2)
            if score < best_rel_score:
                best_rel_score = score
                best_rel = rel 
            if rel_score > score:
                rank = rank+1
        return best_rel, best_rel_score, rank


    # search path between the pair (sub, obj).
    def dfs_search(self, path, path_emb, vis, kb, obj, sub_emb, rel_emb, obj_emb, max_depth):
        cur_v = path[-1][0]
        if cur_v == obj:
            # find a path between sub and obj
            print("path2rel_sim: {1:.6f}, path2obj_score: {2:.6f}, path: {0}".format(path, np.sum((rel_emb-path_emb)**2), np.sum((sub_emb+path_emb-obj_emb)**2)))
            return True

        if len(path) > max_depth:
            return False;
        res = False
        for nbr in kb.get_adj(cur_v):
            if vis[nbr[0]] == True:
                continue
            vis[nbr[0]] = True
            path.append((nbr[0], str(nbr[1])))
            if self.dfs_search(path, path_emb+self.pick_rel(nbr[1]), vis, kb, obj, sub_emb, rel_emb, obj_emb, max_depth):
                vis[nbr[0]] = False # If there is an valid path, we should set vis[nbr[0]] as False to revisited agian.
                res = True
            del path[-1]
            #vis[nbr[0]] = False

        for nbr in kb.get_inv_adj(cur_v):
            if vis[nbr[0]] == True:
                continue
            vis[nbr[0]] = True
            path.append((nbr[0],"inv_"+str(nbr[1])))
            if self.dfs_search(path, path_emb-self.pick_rel(nbr[1]), vis, kb, obj, sub_emb, rel_emb, obj_emb, max_depth):
                vis[nbr[0]] = False
                res = True
            del path[-1]
            #vis[nbr[0]] = False
        return res
