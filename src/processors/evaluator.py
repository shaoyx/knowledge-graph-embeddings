
import numpy as np


BATCHSIZE = 1000


class Evaluator(object):
    def __init__(self, metric, nbest=None, filtered=False, whole_graph=None):
        assert metric in ['mrr', 'hits', 'all'], 'Invalid metric: {}'.format(metric)
        if metric == 'hits':
            assert nbest, 'Please indicate n-best in using hits'
        if filtered:
            assert whole_graph, 'If use filtered metric, Please indicate whole graph'
            self.all_graph = whole_graph
        self.metric = metric
        self.nbest = nbest
        self.filtered = filtered
        self.batchsize = BATCHSIZE
        self.ress = []
        self.id2sub_list = []
        self.id2obj_list = []
        self.sr2o = {}
        self.ro2s = {}

    def run(self, model, dataset):
        if self.metric == 'mrr':
            res = self.cal_mrr(model, dataset)
        elif self.metric == 'hits':
            res = self.cal_hits(model, dataset, self.nbest)
        else:
            raise NotImplementedError
        self.ress.append(res)
        return res

    def run_all_matric(self, model, dataset, knowledge_base):
        """
        calculating MRR, Hits@1,3,10 (raw and filter)
        """
        n_sample = len(dataset)
        sum_rr_raw = 0.
        sum_rr_flt = 0.
        n_corr_h1_raw = 0
        n_corr_h1_flt = 0
        n_corr_h3_raw = 0
        n_corr_h3_flt = 0
        n_corr_h10_raw = 0
        n_corr_h10_flt = 0
        r_corr_h1_raw = 0
        r_corr_h1_flt = 0
        r_corr_h3_raw = 0
        r_corr_h3_flt = 0
        r_corr_h10_raw = 0
        r_corr_h10_flt = 0
        start_id = 0
        correct_triple = 0
        for samples in dataset.batch_iter(self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))
            print("{}/{}".format(start_id, n_sample))
            
            rel_raw_scores = model.cal_rel_scores(subs, objs)
            rel_raw_ranks = self.cal_rank(rel_raw_scores, rels)
            # for idx in range(len(rels)):
            #     print('id:{}, prob:{:.6f}, rank:{}'.format(rels[idx],rel_raw_scores[idx][rels[idx]],rel_raw_ranks[idx]))
            r_sum_rr_raw += sum(float(1/rank) for rank in rel_raw_ranks)
            r_corr_h1_raw += sum(1 for rank in rel_raw_ranks if rank <= 1)
            r_corr_h3_raw += sum(1 for rank in rel_raw_ranks if rank <= 3)
            r_corr_h10_raw += sum(1 for rank in rel_raw_ranks if rank <= 10)

            if model.__class__.__name__ == "LogisticReg" and model.cls_type == "triplet_clssifer":
                correct_triple += model.check_triple(subs, objs, rels)

            # TODO: methods for deep analyze
            # model.analyze(knowledge_base, subs, rels, objs)

            # TODO: partitioned calculation
            # search objects
            raw_scores = model.cal_scores(subs, rels)
            raw_ranks = self.cal_rank(raw_scores, objs)
            # for idx in range(len(objs)):
            #     print('id:{}, prob:{:.6f}, rank:{}'.format(objs[idx],raw_scores[idx][objs[idx]],raw_ranks[idx]))
            sum_rr_raw += sum(float(1/rank) for rank in raw_ranks)
            n_corr_h1_raw += sum(1 for rank in raw_ranks if rank <= 1)
            n_corr_h3_raw += sum(1 for rank in raw_ranks if rank <= 3)
            n_corr_h10_raw += sum(1 for rank in raw_ranks if rank <= 10)
            # filter
            if self.filtered:
                flt_scores = self.cal_filtered_score_fast(subs, rels, objs, ids, raw_scores)
                flt_ranks = self.cal_rank(flt_scores, objs)
                sum_rr_flt += sum(float(1/rank) for rank in flt_ranks)
                n_corr_h1_flt += sum(1 for rank in flt_ranks if rank <=1)
                n_corr_h3_flt += sum(1 for rank in flt_ranks if rank <=3)
                n_corr_h10_flt += sum(1 for rank in flt_ranks if rank <=10)

            # search subjects
            raw_scores_inv = model.cal_scores_inv(rels, objs)
            raw_ranks_inv = self.cal_rank(raw_scores_inv, subs)
            # for idx in range(len(objs)):
            #     print('id:{}, prob:{:.6f}, rank:{}'.format(objs[idx],raw_scores_inv[idx][objs[idx]], raw_ranks_inv[idx]))
            sum_rr_raw += sum(float(1/rank) for rank in raw_ranks_inv)
            n_corr_h1_raw += sum(1 for rank in raw_ranks_inv if rank <= 1)
            n_corr_h3_raw += sum(1 for rank in raw_ranks_inv if rank <= 3)
            n_corr_h10_raw += sum(1 for rank in raw_ranks_inv if rank <= 10)
            # filter
            if self.filtered:
                flt_scores_inv = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, raw_scores_inv)
                flt_ranks_inv = self.cal_rank(flt_scores_inv, subs)
                sum_rr_flt += sum(float(1/rank) for rank in flt_ranks_inv)
                n_corr_h1_flt += sum(1 for rank in flt_ranks_inv if rank <= 1)
                n_corr_h3_flt += sum(1 for rank in flt_ranks_inv if rank <= 3)
                n_corr_h10_flt += sum(1 for rank in flt_ranks_inv if rank <= 10)

            start_id += len(samples)

        return {'MRR': sum_rr_raw/n_sample/2,
                'Hits@1': n_corr_h1_raw/n_sample/2,
                'Hits@3': n_corr_h3_raw/n_sample/2,
                'Hits@10': n_corr_h10_raw/n_sample/2,
                'MRR(filter)': sum_rr_flt/n_sample/2,
                'Hits@1(filter)': n_corr_h1_flt/n_sample/2,
                'Hits@3(filter)': n_corr_h3_flt/n_sample/2,
                'Hits@10(filter)': n_corr_h10_flt/n_sample/2,
                'R_MRR': r_sum_rr_raw/n_sample,
                'R_Hits@1': r_corr_h1_raw/n_sample,
                'R_Hits@3': r_corr_h3_raw/n_sample,
                'R_Hits@10': r_corr_h10_raw/n_sample,
                'Triple prec': correct_triple*1.0/n_sample}

    def cal_mrr(self, model, dataset):
        n_sample = len(dataset)
        sum_rr = 0.
        start_id = 0
        for samples in dataset.batch_iter(self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))
            scores = model.cal_scores(subs, rels)
            if self.filtered:
                scores = self.cal_filtered_score_fast(subs, rels, objs, ids, scores)
            ranks1 = self.cal_rank(scores, objs)

            scores = model.cal_scores_inv(rels, objs)
            if self.filtered:
                scores = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, scores)
            ranks2 = self.cal_rank(scores, subs)
            sum_rr += sum(float(1/rank) for rank in ranks1 + ranks2)
            start_id += len(samples)
        return float(sum_rr/n_sample/2)

    def cal_hits(self, model, dataset, nbest):
        n_sample = len(dataset)
        n_corr = 0
        start_id = 0
        for samples in dataset.batch_iter(self.batchsize, rand_flg=False):
            subs, rels, objs = samples[:, 0], samples[:, 1], samples[:, 2]
            ids = np.arange(start_id, start_id+len(samples))
            scores = model.cal_scores(subs, rels)
            if self.filtered:
                scores = self.cal_filtered_score_fast(subs, rels, objs, ids, scores)
            res = np.flip(np.argsort(scores), 1)[:, :nbest]
            n_corr += sum(1 for i in range(len(objs)) if objs[i] in res[i])

            scores = model.cal_scores_inv(rels, objs)
            if self.filtered:
                scores = self.cal_filtered_score_inv_fast(subs, rels, objs, ids, scores)
            res = np.flip(np.argsort(scores), 1)
            n_corr += sum(1 for i in range(len(subs)) if subs[i] in res[i])
            start_id += len(samples)
        return float(n_corr/n_sample/2)

    def cal_filtered_score_fast(self, subs, rels, objs, ids, raw_scores, metric='sim'):
        assert metric in ['sim', 'dist']
        new_scores = []
        for s, r, o, i, score in zip(subs, rels, objs, ids, raw_scores):
            true_os = self.id2obj_list[i]
            true_os_rm_o = np.delete(true_os, np.where(true_os == o))
            if metric == 'sim':
                score[true_os_rm_o] = -np.inf
            else:
                score[true_os_rm_o] = np.inf
            new_scores.append(score)
        return new_scores

    def cal_filtered_score_inv_fast(self, subs, rels, objs, ids, raw_scores, metric='sim'):
        assert metric in ['sim', 'dist']
        new_scores = []
        for s, r, o, i, score in zip(subs, rels, objs, ids, raw_scores):
            true_ss = self.id2sub_list[i]
            true_ss_rm_s = np.delete(true_ss, np.where(true_ss==s))
            if metric == 'sim':
                score[true_ss_rm_s] = -np.inf
            else:
                score[true_ss_rm_s] = np.inf
            new_scores.append(score)
        return new_scores

    def cal_rank(self, score_mat, ents):
        # print(score_mat)
        # print(ents)
        return [np.sum(score >= score[e]) for score, e in zip(score_mat, ents)]

    def get_best_info(self):
        if self.metric == 'mrr' or self.metric == 'hits' or self.metric == 'acc':  # higher value is better
            best_val = max(self.ress)
        elif self.metric == 'mr':
            best_val = min(self.ress)
        else:
            raise ValueError('Invalid')
        best_epoch = self.ress.index(best_val) + 1
        return best_epoch, best_val

    def prepare_valid(self, dataset):
        for i in range(len(dataset)):
            s, r, o = dataset[i]
            os = self.all_graph.search_obj_id(s, r)
            ss = self.all_graph.search_sub_id(r, o)
            self.id2obj_list.append(os)
            self.id2sub_list.append(ss)
            self.sr2o[(s, r)] = os
            self.ro2s[(r, o)] = ss

