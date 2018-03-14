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
        
        with open(args.kb) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                sub_emb = model.pick_ent(ent_vocab[sub])
                obj_emb = model.pick_ent(ent_vocab[obj])
                rel_id = rel_vocab[rel]
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
                X.append(sub_emb-obj_emb)
                y.append(rel_id)

    return np.array(X), np.array(y)

def inspect(args):
    X, y = read_data(args.data, args.model, args)
    if args.source == "WN18" and args.model == "deepwalk":
        y = y >= 40943 # label 1 for relation, label 0 for entity.
    TwoDimProjection(X, y, args.source+"_"+args.model).visualize()

if __name__ == '__main__':
    p = argparse.ArgumentParser('Inspector for data analysis')
    p.add_argument('--data', type=str, help='the path of high-dimensional data')
    p.add_argument('--source', default="WN18", type=str, help='the source of data')
    p.add_argument('--model', default="deepwalk", type=str, help='the description of visualizing approach')
    p.add_argument('--ent', type=str, help='entity id list')
    p.add_argument('--rel', type=str, help='relation id list')
    p.add_argument('--kb', type=str, help='the knowledge base data')

    args = p.parse_args()

    inspect(args)
