import math

# only continuous attributes

class DecisionTreeClassifier():
    def entropy(self, D):
        class_labels_counts = D.iloc[:,-1].value_counts().to_dict()

        entropy = 0
        for val in class_labels_counts.values():
            prob = val/len(D)
            entropy += prob * math.log2(prob)
        return -entropy
    
    def findBestSplit(self, D, a):
        p0 = self.entropy(D)
        gains = dict.fromkeys(sorted(list(D[a].unique())), 0)

        for val in gains.keys():
            left = D[D[a] <= val]
            right = D[D[a] > val]
            gains[val] = p0 - len(left)/len(D) * self.entropy(left) - len(right)/len(D) * self.entropy(right)
        max_value = max(gains, key=gains.get)
        max_gain = gains[max_value]
        return max_value, max_gain
    
    def selectSplittingAttribute(self, D, A, threshold): # uses information gain
        gains = dict.fromkeys(A, 0)

        for a in A:
            gains[a] = self.findBestSplit(D, a)[1]
        best = max(gains, key=gains.get)
        if gains[best] > threshold:
            return best
        else:
            return None
        
    def C45(self, D, A, og_D, threshold, parent_val=None, parent_var=None, sign=None):
        T = {}
        class_labels_counts = D.iloc[:,-1].value_counts().to_dict()
        c = max(class_labels_counts, key=class_labels_counts.get)

        # Step 1: check termination conditions
        if len(class_labels_counts.keys()) == 1:
            T['decision'] = list(class_labels_counts.keys())[0]
            if sign == '<=':
                T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
            else:
                T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])

        elif A == []:
            T['decision'] = c
            if sign == '<=':
                T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
            else:
                T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
        # Step 2: select splitting attribute
        else: 
            best = self.selectSplittingAttribute(D, A, threshold)
            if best is None: # no attribute is good for a split
                if (parent_val is None) and (parent_var is None):
                    print('Threshold value too high to even select a root node')
                    exit()
                T['decision'] = c
                if sign == '<=':
                    T['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
                else:
                    T['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
            # Step 3: tree construction
            else:
                T['var'] = best
                T['edges'] = []
                v = self.findBestSplit(D, best)[0]
                for i in range(2):
                    T['edges'].append({})
                    T['edges'][i]['edge'] = {}
                    if i == 0:
                        D_v = D[D[best] <= v]
                        sign = '<='
                    else:
                        D_v = D[D[best] > v]
                        sign = '>'
                    T['edges'][i]['edge']['value'] = sign + ' ' + str(v)
                    if not D_v.empty:
                        T_v = self.C45(D_v, list(D_v.columns)[:-1], og_D, threshold, v, best, sign) # recursive call
                        if 'decision' in T_v.keys():
                            T['edges'][i]['edge']['leaf'] = T_v
                        else:
                            T['edges'][i]['edge']['node'] = T_v
                    else: # ghost leaf node
                        T['edges'][i]['edge']['leaf']['decision'] = c
                        if sign == '<=':
                            T['edges'][i]['edge']['leaf']['p'] = og_D[og_D[parent_var] <= parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] <= parent_val])
                        else:
                            T['edges'][i]['edge']['leaf']['p'] = og_D[og_D[parent_var] > parent_val].iloc[:,-1].value_counts()[T['decision']]/len(og_D[og_D[parent_var] > parent_val])
        return T

# if __name__ == '__main__':
#     csv_file = sys.argv[1]
#     before = csv_file.split('.csv')[0]
#     dataset = before.split('/')[-1]

#     # getting values
#     D = pd.read_csv(csv_file)
#     og_D = D.copy()

#     A = list(D.columns)[:-1]

#     # run C4.5
#     classifier = DecisionTreeClassifier()
#     T = classifier.C45(D, A, og_D, threshold=0.18) # heart
#     T = {'dataset': csv_file, 'node': T}
#     with open(dataset + '.json', 'w') as outfile:
#         json.dump(T, outfile, indent=4)