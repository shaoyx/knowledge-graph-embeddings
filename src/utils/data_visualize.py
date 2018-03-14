from sklearn.manifold import TSNE

# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

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

        plt.scatter(X[:, 0], X[:, 1], s=1, c=y, alpha=0.6, cmap=plt.cm.Spectral)
        if flag == "save":
            plt.savefig(fname=self.name+".pdf")
        else:
            plt.show()

class TwoDimProjection(BasicVisualizer):
    def __init__(self, data, label, name):
        super(TwoDimProjection, self).__init__(data, label, name)

    def visualize(self):
        X_emb = TSNE(perplexity=100, n_iter=5000).fit_transform(self.X)
        self.scatter_plot(X_emb, self.y)
