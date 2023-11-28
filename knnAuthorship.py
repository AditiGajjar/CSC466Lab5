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
