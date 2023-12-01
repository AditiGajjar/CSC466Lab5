import pandas as pd
import json
import sys

def classify(data, T):
    # reached a leaf node, return the classification result
    if 'decision' in T:
        return T['decision']
    
    if 'var' in T:
        value = data.get(T['var'])
    
    if 'edges' in T:
        for dic in T['edges']:
            if dic['edge']['value'] == value:
                return classify(data, dic)
            elif dic['edge']['value'].split()[0] == '<=' and value <= float(dic['edge']['value'].split()[1]):
                return classify(data, dic)
            elif dic['edge']['value'].split()[0] == '>' and value > float(dic['edge']['value'].split()[1]):
                return classify(data, dic)
                
    if ('edge' in T) and ('leaf' in T['edge']):
        return classify(data, T['edge']['leaf'])
    
    if ('edge' in T) and ('value' in T['edge']):
        return classify(data, T['edge']['node'])
            
if __name__ == '__main__':
    input = pd.read_csv(sys.argv[1])
    ground_truth = pd.read_csv(sys.argv[2])

    y_pred = input.iloc[:,1]
    y = ground_truth.iloc[:,1]
    author_dict = dict.from_keys(y.unique(), dict.from_keys(['hits', 'strikes', 'misses'], 0))
    correct = 0
    incorrect = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            author_dict[y[i]]['hits'] += 1
        else:
            author_dict[y_pred[i]]['strikes'] += 1
    
    for stats in author_dict.values():
        stats['misses'] = 50 - stats['hits']

    # knn: precision, recall, f1

    print('Total number of documents with correctly predicted authors: ' + str(correct))
    print('Total number of documents with incorrectly predicted authors: ' + str(incorrect))
    print('Overall accuracy: ' + str(correct/len(y)))
    cf_matrix = pd.crosstab(y, y_pred, rownames=['Actual'], colnames=['Predicted'])

# python3 classifierEvaluation.py

# for each author in the dataset, output the total number of hits (correctly predicted), strikes (false positives predicted),
# misses (document written by author, which were not attributed to the author) = 50 - hits
# precision, recall and f1 of knn

# 50x50 confusion matrix dump into csv file
# author names for first row and first col