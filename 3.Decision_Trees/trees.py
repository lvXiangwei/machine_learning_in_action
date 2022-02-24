from doctest import Example
from math import log
import operator

def calc_shannnon_entropy(dataset):
    num_entries = len(dataset)
    labels_count = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        labels_count[current_label]=labels_count.get(current_label,0)+1
    
    shannon_ent = 0.0
    for key in labels_count:
        prob = labels_count[key]/num_entries
        shannon_ent += - prob * log(prob, 2)
    return shannon_ent

def create_dataset():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            ret_dataset.append(feat_vec[:axis]+feat_vec[axis+1:])
    return ret_dataset

def choose_best_feature_to_split(dataset):
    num_features = len(dataset[0])-1
    base_entropy = calc_shannnon_entropy(dataset)

    best_info_gain =0.0
    best_feature = -1

    for i in range(num_features):
        feat_set = {example[i] for example in dataset}
        new_entropy = 0.0
        for value in feat_set:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/len(dataset)
            new_entropy += prob * calc_shannnon_entropy(sub_dataset)
        info_gain = base_entropy - new_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1

    sorted_class_count = sorted(class_count.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]

def create_tree(dataset, labels):
    class_list  = [example[-1] for example in dataset]
    #1. stop when all classes are equal
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    #2. when no more features, return majority
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    best_feature = choose_best_feature_to_split(dataset)
    #print(best_feature, labels)
    best_feature_label = labels[best_feature]
    
    my_tree = {best_feature_label:{}} # use nested dictionaries to represent a tree
    del labels[best_feature]

    vals_set = {example[best_feature] for example in dataset}

    for val in vals_set:
        sub_labels = labels[:] # copy all of labels, so trees don't mess up existing labels
        
        my_tree[best_feature_label][val] = create_tree(split_dataset(dataset, best_feature, val), sub_labels)
    return my_tree
    