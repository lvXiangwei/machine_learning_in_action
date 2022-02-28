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

def bag_of_words2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec

def train_naive_Bayes0(train_mat, train_category):
    '''
    train_mat: (number of sentences, number of vocab)
    train_category: (number of sentences, ) 1 indicates abusive
    '''     

    num_train_docs, num_of_vocab = len(train_mat), len(train_mat[0])
    p_abusive = sum(train_category) / num_train_docs

    # p0_num = np.zeros(num_of_vocab)
    # p1_num = np.zeros(num_of_vocab)
    p0_num = np.ones(num_of_vocab)
    p1_num = np.ones(num_of_vocab)
    # p0_denom = p1_denom = 0.0
    p0_denom = p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_mat[i]
            p1_denom += sum(train_mat[i])
        else:
            p0_num += train_mat[i]
            p0_denom += sum(train_mat[i])
    # p1_vec = p1_num / p1_denom
    # p0_vec = p0_num / p0_denom
    p1_vec = np.log(p1_num / p1_denom)
    p0_vec = np.log(p0_num / p0_denom)
    return p0_vec, p1_vec, p_abusive    

def classify_naive_Bayes(vec_2_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_2_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec_2_classify * p0_vec) + np.log(1 - p_class1)

    return 1 if p1 > p0 else 0 

def test_naive_Bayes():
    dataset, labels = load_dataset()
    voc = create_vocab_list(dataset)
    train_mat = []
    for doc in dataset:
         train_mat.append(set_of_words2vec(voc, doc))
    p0_vec, p1_vec, p_abusive =  train_naive_Bayes0(np.array(train_mat),np.array(labels))

    test_entry = ['love' , 'my', 'dalmation']
    this_doc = np.array(set_of_words2vec(voc, test_entry))
    print(f'{test_entry} is classifed as: {classify_naive_Bayes(this_doc, p0_vec, p1_vec, p_abusive)}')

    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words2vec(voc, test_entry))
    print(f'{test_entry} is classifed as: {classify_naive_Bayes(this_doc, p0_vec, p1_vec, p_abusive)}')


def text_parse(origin_str):
    import re
    list_of_tokens = re.split(r'\W+', origin_str)
    return [token.lower() for token in list_of_tokens if len(token) > 2]


def spam_test():
    import random
    doc_list = []
    class_list = []

    for i in range(1, 26):
 
        word_list = text_parse(open(f'email/spam/{i}.txt').read())
        doc_list.append(word_list)
        class_list.append(1)

  
        word_list = text_parse(open(f'email/ham/{i}.txt').read())
        doc_list.append(word_list)
        class_list.append(0)
    
    vocab_list = create_vocab_list(doc_list)

    train_set = list(range(50))
    test_set = []

    for i in range(10):
        rand_idx = int(random.uniform(0, len(train_set)))
        test_set.append(train_set[rand_idx])
        del train_set[rand_idx]

    train_mat = []
    train_classes = []
    for doc_idx in train_set:
        train_mat.append(set_of_words2vec(vocab_list, doc_list[doc_idx]))
        train_classes.append(class_list[doc_idx])
    p0_vec, p1_vec, p_spam = train_naive_Bayes0(train_mat, train_classes)
    
    error_count = 0
    for doc_idx in test_set:
        test_word_vec = set_of_words2vec(vocab_list, doc_list[doc_idx])
        if classify_naive_Bayes(test_word_vec, p0_vec, p1_vec, p_spam) != class_list[doc_idx]:
            error_count += 1

    print(f'the error rate is:{error_count/len(test_set)}')
