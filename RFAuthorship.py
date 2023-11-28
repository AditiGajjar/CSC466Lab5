# input:
    # 1. the file of vectorized document representations,
    # 2. a flag indicating the similarity metric to be used in the computation
    # 3. values for 
        # (a) number of decision trees to build
        # (b) number of attributes in a decision tree
        # (c) number of data points to be used to construct a decision tree
        # (d) any parameters you need for the C45 algorithm itself (e.g., the threshold)
# output:
    # authorship label predicted for each of the documents in the Reuters 50-50 dataset
# notes:
    # no need for cross validation

# Approach 1: 50-class classification problem. 
# In this case, you run a single classification task that has a class variable consisting of 50 labels (individual authors)

