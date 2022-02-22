import numpy as np
import operator

def create_dataset():
    group = np.array([[1.0,1.1],[1.0,1.0],\
        [0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # 1.Distance calculation
    diff_mat = inX - dataSet
    sq_diff_mat = diff_mat**2
    sq_distances = np.sum(sq_diff_mat, axis=1)
    sort_dist_indicies = sq_distances.argsort()
    class_count={}
    
    # 2.Voting with lowest k distances
    for i in range(k):
        vote_i_label = labels[sort_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label,0)+1

    # 3.Sort dictionary
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class_count[0][0]

def file2matrix(filename):
    with open(filename) as f:
        array_of_lines = f.readlines()
    no_of_lines = len(array_of_lines)

    return_mat = np.zeros((no_of_lines,3))
    class_label_vector = []

    for i in range(no_of_lines):
        line = array_of_lines[i]
        list_from_line = line.split('\t')
        return_mat[i,:] = list_from_line[:3]
        class_label_vector.append(int(list_from_line[-1]))

    return return_mat, class_label_vector

def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = (dataset - min_vals) / ranges
    #need ranges, min_vlas to normalizie test data
    return norm_dataset, ranges, min_vals