import numpy as np

def load_dataset():
    postingList=[['my', 'dog', 'has', 'flea', \
    'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', \
    'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', \
    'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
    'to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList,classVec

def create_vocab_list(dataset):
    vocab_set = {word for document in dataset for word in document}
    return list(vocab_set)

def set_of_words2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print(f'the word {word} is not in my Vocabulary!')
    return return_vec

def train_naive_Bayes0(train_mat, train_category):
    '''
    train_mat: (number of sentences, number of vocab)
    train_category: (number of sentences, ) 1 indicates abusive
    '''     

    num_train_docs, num_of_vocab = len(train_mat), len(train_mat[0])
    p_abusive = sum(train_category) / num_train_docs

    p0_num = np.zeros(num_of_vocab)
    p1_num = np.zeros(num_of_vocab)
    p0_denom = p1_denom = 0.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])
    p1_vec = p1_num / p1_denom
    p0_vec = p0_num / p0_denom

    return p0_vec, p1_vec, p_abusive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    