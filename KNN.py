import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt


def KNN_classify(k,X_train,y_train,x):

    assert 1<=k<=X_train.shape[0],"k must be valid"
    assert  X_train.shape[0] == y_train.shape[0],\
    "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0],\
    "the feature number of x must be equal to X_train"


    distances=[sqrt(np.sum(x_train-x)**2) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes =Counter(topK_y)
    return  votes.most_common(1)[0][0]






class KNNClassifier:
    def __init__(self,k):
        """initialize knn classifier"""
        assert  k>=1,"k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self,X_train,y_train):
        """according to training data
        train the KNN classifier

         """
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must equal to the size of y_train"
        assert  self.k <= X_train.shape[0],\
        "the size of X_train must be at least k"
        self._X_train =X_train
        self._y_train =y_train

        return  self
    def predict(self,X_predict):
        assert self._X_train is not None and self._y_train is not None,\
        "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给定单个待预测数据x, 返回x的预测结果"""
        assert self._X_train.shape[1] == x.shape[0], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum(x_train - x) ** 2) for x_train \
                     in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "KNN(k=%d)" % self.k