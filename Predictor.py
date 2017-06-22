import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from Bio.SubsMat import MatrixInfo
blosum = MatrixInfo.blosum62

def score_match(pair, matrix):
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]

def switch(seq1, consense):
    switch = []
    for i in range(len(seq1)):
        pair = (seq1[i], consense[i])
        switch += [score_match(pair, blosum)]
    return switch


with open("project_training.txt","r") as lines:
    content = lines.read().splitlines()

header = content[0].split('\t') + ['Pos_%s'%i for i in range(0,9)]
peptides = []
consense = list('AQIDNYNKF')

for line in content[1:]:
    tmp = line.split('\t')
    AAPos = switch(tmp[0], consense)
    peptides = peptides + [tmp + AAPos]

peptides = np.array(peptides)

X = np.array(peptides[:, 3:5], dtype = int)
print(np.shape(X))
cls = np.array(peptides[:, 2], dtype = int)
print(np.shape(cls))

kernel = 1.0 * RBF([1.0])
gpc_rbf_isotopic = GaussianProcessClassifier(kernel = kernel)
fitted_gpc_rbf_isotropic = gpc_rbf_isotopic.fit(X, cls)

def plot_predictions(clsfr, title, h=.02):
    # Plot the predicted probabilities. For that, we will assign a color to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clsfr.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y])
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" %
              (title, clsfr.log_marginal_likelihood(clsfr.kernel_.theta)))

    plt.tight_layout()
    plt.show()

plot_predictions(fitted_gpc_rbf_isotropic, 'isotropic')

print("Done")