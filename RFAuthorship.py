import pandas as pd
import sys
import numpy as np
import random
from InduceC45 import DecisionTreeClassifier
from classifierEvaluation import classify

class RFClassifier():
    def __init__(self, m, k, N, threshold):
        self.m = m
        self.k = k
        self.N = N
        self.threshold = threshold
       
    def data_selection(self, training):
        cols = random.sample(list(training.columns)[:-1], self.m) # without replacement
        cols.append(list(training.columns)[-1])
        rows = list(np.random.choice(range(len(training)), size=self.k, replace=True)) # with replacement
        return training.loc[rows, cols]

    def random_forest(self):
        forest = [] # list that contains all the trees
        for i in range(self.N):
            tree_data = self.data_selection(D)
            og_tree_data = tree_data.copy()
            classifier = DecisionTreeClassifier()
            tree = classifier.C45(tree_data, list(tree_data.columns)[1:], og_tree_data, threshold)
            forest.append(tree)

        y_pred = []
        for i in range(len(D)):
            tree_preds = []
            for tree in forest:
                pred = classify(D.iloc[i, 1:].to_dict(), tree)
                tree_preds.append(pred)
            most_common_pred = max(set(tree_preds), key=tree_preds.count)
            y_pred.append(most_common_pred)

        res = pd.DataFrame({'doc': D.iloc[:,0], 'author': y_pred})
        res.to_csv('authors_rf.csv', index=False)

if __name__ == '__main__':
    D = pd.read_csv(sys.argv[1])
    m = int(sys.argv[2])
    k = int(sys.argv[3])
    N = int(sys.argv[4])
    threshold = float(sys.argv[5])

    # check that input parameters are valid
    if m > len(list(D.columns)) - 1:
        print('m is larger than number of attributes, please provide a smaller m')
    if k > len(D):
        print('k is larger than number of data points, please provide a smaller k')

    classifier = RFClassifier(m, k, N, threshold)
    classifier.predict()

# python3 RFAuthorship.py tfidf_vectorized_docs.csv m k N threshold
# python3 RFAuthorship.py tfidf_vectorized_docs.csv 50 25 50 0.05