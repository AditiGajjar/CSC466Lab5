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
    def __init__(self, D, k, distance_metric): 
        self.D = D
        self.k = k
        self.distance_metric = distance_metric
    
    # document frequency - # of documents in which term ti occurs
    def doc_freq(matrix):
        cols = matrix.shape[1]
        temp = np.zeros(cols)
        for i in range(cols):
            temp[i] = np.count_nonzero(matrix[:,i])
        return temp

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = dot_product / (norm_a * norm_b)
        return 1 - similarity

    def okapi(self, matrix, dj):
        # dj is ground truth file
        rows = matrix.shape[0] # docs
        cols = matrix.shape[1] # words
        #distance = np.zeros((rows, rows)) # matrix with all the distances
        
        # n = len(matrix)
        dlj = len(dj) # length of document
        avdl = len(dj).mean()  # average length of document
        k1 = 1.5 # usually between 1.0 - 2.0 (normalization parameter for dj)
        b = 0.75 # usually 0.75 (normalization parameter for doc length)
        k2 = 500 # usually between 1-1000 (normalization parameter for query q)
        dfi = self.doc_freq(matrix) # 

        for q in range(rows): # query is each row
            for j in range(rows):
                if j != q:
                    sum = 0
                    for i in range(cols):
                        dfi[i]
                        x = np.log(rows - dfi[i])
                        y = ((k1 + 1) * matrix[j,i]) / (k1 * (1 - b + (b * (dlj[j]/avdl))))
                        z = ((k2 + 1)* matrix[q,i]) / (k2 * matrix[q, i])
                        sum += x * y * z
                    #distance[q,j] = sum
        return sum

                    

    def calculate_distance_matrix(self, matrix, distance_metric, gt):
        rows = matrix.shape[0] # docs
        cols = matrix.shape[1] # words
        distance = np.zeros((rows, rows))
        if self.distance_metric == 'Cosine similarity':
            for i in range(rows):
                for j in range(i+1, rows):
                    return self.cosine_similarity(matrix[i], matrix[j])
        elif self.distance_metric == 'Okapi':
            return self.okapi(matrix, gt)

    #old knn
    def knn(self):
        class_col = self.D.pop(class_name)
        self.D[class_name] = class_col 

        preds = []
        file_name = dataset + '_knn.txt'
        with open(file_name, 'w') as file:
            file.write(' '.join(list(D.columns)) + '\n')
            for i in range(len(self.D)): # rows 
                file.write(' '.join(map(str, self.D.iloc[i, :-1])) + ' ')
                distances = {}
                for j in range(len(self.D)): # cols
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
    # from command line
    if len(sys.argv) != 3:
        print('Wrong format')
        sys.exit(1)

    csv_file = sys.argv[1]
    before = csv_file.split('.csv')[0]
    dataset = before.split('/')[-1]
    D = pd.read_csv(csv_file)
    class_name = list(D.columns)[-1]
    
    k = int(sys.argv[2])

    choice = input('Enter distance metric to use for numeric attributes: ')
    if choice not in ['Cosine similarity','Okapi']:
        print('Please choose a valid distance metric')
        sys.exit(1)

    classifier = KNNClassifier(D, k, choice)
    classifier.knn()