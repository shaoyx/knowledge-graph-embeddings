import numpy as np
import argparse

from gensim.models import KeyedVectors

from utils.data_visualize import *
from utils.dataset import *

def read_data(path, data_type, args):
    X = []
    y = []
    if data_type == 'deepwalk':
        word_vector = KeyedVectors.load_word2vec_format(path, binary=False)
        for word in word_vector.vocab:
            vector = word_vector[word]
            X.append(vector)
            y.append(int(word))
    if data_type == "transe":
        from models.transe import TransE as Model
        model = Model.load_model(path)
        ent_vocab = Vocab.load(args.ent)
        rel_vocab = Vocab.load(args.rel)

        valid_rels = np.random.randint(0, len(rel_vocab), args.n_cls)
        
        with open(args.kb) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_emb = model.pick_ent(ent_vocab[sub])
                obj_emb = model.pick_ent(ent_vocab[obj])
                rel_id = rel_vocab[rel]
                if (args.n_cls >= len(rel_vocab)) or (rel_id in valid_rels):
                    X.append(sub_emb-obj_emb)
                    y.append(rel_id)
    else:
        word_vector = KeyedVectors.load_word2vec_format(path, binary=False)
        ent_vocab = Vocab.load(args.ent)
        rel_vocab = Vocab.load(args.rel)

        def pick_emb(eid):
            if str(eid) in word_vector:
                return word_vector[str(eid)]
            else:
                return np.array([0.0 for i in range(len(word_vector[0]))])

        with open(args.kb) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_emb = pick_emb(ent_vocab[sub])
                obj_emb = pick_emb(ent_vocab[obj])
                rel_id = rel_vocab[rel]
                valid_rels = np.random.randint(0, len(rel_vocab), args.n_cls)
                if (args.n_cls >= len(rel_vocab)) or (rel_id in valid_rels):
                    X.append(sub_emb-obj_emb)
                    y.append(rel_id)

    return np.array(X), np.array(y)

def inspect(args):
    X, y = read_data(args.data, args.model, args)
    
    if args.source == "WN18" and args.model == "deepwalk":
        y = y >= 40943 # label 1 for relation, label 0 for entity.
    if args.vis_type == "pairwise":
        s = len(set(y.tolist()))

        def partX(X, y):
            xx = {}
            for idx in range(len(X)):
                key = y[idx]
                if key in xx:
                    xx[key] = np.vstack((xx[key], X[idx]))
                else:
                    xx[key] = np.reshape(X[idx], (1, len(X[idx])))
            return xx

        part_X = partX(X, y)
        # for idx in range(s):
        #     print("idx:{} len:{}".format(idx, len(part_X[idx])))
        # print("End")
        for i in range(s):
            for j in range(i+1, s):
                # print(part_X[i].shape)
                # print(part_X[j].shape)
                a = np.vstack((part_X[i], part_X[j]))
                xiy = np.array([i for k in range(len(part_X[i]))])
                xjy = np.array([j for k in range(len(part_X[j]))])
                b = np.hstack((xiy, xjy))
                # print("i:{},j:{}".format(i,j))
                # print(a.shape)
                # print(b.shape)
                TwoDimProjection(a,b,args.source+"_"+args.model+"_"+str(i)+"_"+str(j)).visualize()

    else:
        TwoDimProjection(X, y, args.source+"_"+args.model).visualize()

if __name__ == '__main__':
    p = argparse.ArgumentParser('Inspector for data analysis')
    p.add_argument('--data', type=str, help='the path of high-dimensional data')
    p.add_argument('--source', default="WN18", type=str, help='the source of data')
    p.add_argument('--model', default="deepwalk", type=str, help='the description of visualizing approach')
    p.add_argument('--ent', type=str, help='entity id list')
    p.add_argument('--rel', type=str, help='relation id list')
    p.add_argument('--kb', type=str, help='the knowledge base data')
    p.add_argument('--n_cls', default=2, type=int, help='the number of different classes')
    p.add_argument('--vis_type', default="all", type=str, help='the type for visualization')

    args = p.parse_args()

    inspect(args)
