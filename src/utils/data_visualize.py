from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

class BasicVisualizer(object):
    def __init__(self, data, label, name_):
        self.X = data
        self.y = label
        self.name = name_

    def visualize(self):
        raise NotImplementedError

    # simple scatter plot
    def scatter_plot(self, X, y, flag="save"):
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        plt.scatter(X[:, 0], X[:, 1], c=y)
        if flag == "save":
            plt.savefig(fname=self.name+".pdf")
        else:
            plt.show()

class TwoDimProjection(BasicVisualizer):
    def __init__(self, data, label, name):
        super(TwoDimProjection, self).__init__(data, label, name)

    def visualize(self):
        X_emb = TSNE().fit_transform(self.X)
        self.scatter_plot(proj, self.y)
