"""Build indexing using extracted features
"""

import numpy as np
import os, os.path
import pickle
import scipy.spatial
from scipy.spatial.distance import pdist, squareform

"""Build index
Given feature matrix feature_mat,
k is the number of nearest neighbors to choose
"""
def build_index(feature_mat, k):
    sample_count_ = feature_mat.shap(0)
    index = np.zeros((sample_count_, k), dtype=numpy.int)
    for idx in range(sample_count_):
        feature = feature_mat[idx, ...]
        dist =
