import numpy as np
import operator
from os import listdir


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

def dating_classtest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = auto_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwriting_class_test():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(mTest)))