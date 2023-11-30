# input:
    # 1. the file of vectorized document representations,
    # 2. a flag indicating the similarity metric to be used in the computation
    # 3. a value of k (number of nearest neighbors to check)
# output:
    # authorship label predicted for each of the documents in the Reuters 50-50 dataset
# notes:
    # must use  all-but-one validation

# Approach 1: 50-class classification problem. 
# In this case, you run a single classification task that has a class variable consisting of 50 labels (individual authors).

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import sys

class KNNClassifier():
    def __init__(self, D, k, similarity_metric): 
        self.D = D
        self.k = k
        self.distance_metric = similarity_metric # 1 for cosine similarity, 2 for okapi
 
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def manhattan_distance(self, a, b):
        return np.sum(np.abs(a - b))

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return 1 - similarity

    def calculate_distance(self, a, b):
        if self.distance_metric == 'Euclidean':
            return self.euclidean_distance(a, b)
        elif self.distance_metric == 'Manhattan':
            return self.manhattan_distance(a, b)
        elif self.distance_metric == 'Cosine similarity':
            return self.cosine_similarity(a, b)

    def knn(self):
        cate_cols = []
        for col in list(self.D.columns)[:-1]:
            if not is_numeric_dtype(self.D[col]):
                cate_cols.append(col)
        self.D = pd.get_dummies(self.D, columns=cate_cols, dtype=int)
        class_col = self.D.pop(class_name)
        self.D[class_name] = class_col 

        preds = []
        file_name = dataset + '_knn.txt'
        with open(file_name, 'w') as file:
            file.write(' '.join(list(D.columns)) + '\n')
            for i in range(len(self.D)):
                file.write(' '.join(map(str, self.D.iloc[i, :-1])) + ' ')
                distances = {}
                for j in range(len(self.D)):
                    if i != j:
                        val1 = self.D.iloc[i, :-1]
                        val2 = self.D.iloc[j, :-1]
                        dist = self.calculate_distance(val1, val2)
                        distances[j] = dist

                knn = dict(sorted(distances.items(), key=lambda x: x[1])[:self.k])

                neighbor_indices = list(knn.keys())
                neighbor_class_preds = [self.D.iloc[index, -1] for index in neighbor_indices]
                most_common_pred = max(set(neighbor_class_preds), key=neighbor_class_preds.count)
                file.write(str(most_common_pred) + '\n')
                preds.append(most_common_pred)

            actual = self.D.iloc[:,-1]
            cf_matrix = pd.crosstab(actual, preds, rownames=['Actual'], colnames=['Predicted']) 
            
            file.write('\n' + 'Overall confusion matrix: ' + '\n')
            file.write(str(cf_matrix) + '\n')
            i = 0
            correct = 0
            total = 0
            for label in cf_matrix.index:
                TP = cf_matrix.iloc[i, i]
                FP = np.sum(cf_matrix.iloc[:, i]) - TP 
                FN = np.sum(cf_matrix.iloc[i, :]) - TP
                TN = cf_matrix.values.sum() - TP - FP - FN
                correct += TP + TN
                total += TP + FP + FN + TN
                file.write('\n' + "For class label '" + str(label) + "': " + '\n')
                file.write('Confusion matrix: ' + '\n')
                temp = [[TP, FP], 
                        [FN, TN]]
                i += 1
                for row in temp:
                    file.write(str(row) + '\n')
            file.write('\n' + 'Accuracy: ' + str(correct/total) + '\n')

if __name__ == '__main__':
    csv_file = sys.argv[1]
    before = csv_file.split('.csv')[0]
    dataset = before.split('/')[-1]
    D = pd.read_csv(csv_file)
    class_name = list(D.columns)[-1]
    
    k = int(sys.argv[2])

    choice = input('Enter distance metric to use for numeric attributes: ')
    if choice not in ['Euclidean', 'Manhattan', 'Cosine similarity']:
        print('Please choose a valid distance metric')
        sys.exit(1)

    classifier = KNNClassifier(D, k, choice)
    classifier.knn()
    
# python3 knn.py iris.csv 25 Cosine similarity
# python3 knn.py heart_new.csv 25 Manhattan
# python3 knn.py credit.csv 25 Manhattan

# python3 knn.py file flag k
