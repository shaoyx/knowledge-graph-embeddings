import numpy as np
import argparse

from gensim.models import KeyedVectors

from utils.data_visualize import *

def read_data(path, data_type):
    X = []
    y = []
    if data_type == 'deepwalk':
        word_vector = KeyedVectors.load_word2vec_format(path, binary=False)
        for word in word_vector.vocab:
            vector = word_vector[word]
            X.append(vector)
            y.append(int(word))
    return np.array(X), np.array(y)

def inspect(args):
    X, y = read_data(args.data, args.model)
    if args.source == "WN18" and args.model == "deepwalk":
        y = y >= 40943 # label 1 for relation, label 0 for entity.
    TwoDimProjection(X, y).visualize()

if __name__ == '__main__':
    p = argparse.ArgumentParser('Inspector for data analysis')
    p.add_argument('--data', type=str, help='the path of high-dimensional data')
    p.add_argument('--source', default="WN18", type=str, help='the source of data')
    p.add_argument('--model', default="deepwalk", type=str, help='the source of data')

    args = p.parse_args()

    inspect(args)
