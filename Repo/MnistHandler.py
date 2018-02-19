import _pickle as cPickle
import gzip
import numpy as np

def LoadDataFile(filePath, dimensions = 785, labels = False):
    ''' This function is to read the data files
        todo: Add comments, verify
    '''
    temp = np.loadtxt(open(filePath, "rb"), delimiter=",", skiprows=1)
    iD = list(map(int, temp[:, 0]))
    vector = temp[:, 1 : dimensions]
    if labels is True:
        label = list(map(int, temp[:, dimensions]))
        maxColumns = np.max(label) + 1
        oneHot = np.eye(maxColumns)[label]
        returnData = [vector, label]
    else:
        returnData = [iD, vector]
    del temp
    return returnData

def DataLoadMaster(trainDataPath, testDataPath, validationDataPath):
    ''' todo: Add comments '''
    trainData = LoadDataFile(trainDataPath, labels = True)
    testData = LoadDataFile(testDataPath, labels=True)
    validationData = LoadDataFile(validationDataPath, labels = True)
    # testData = None
    # validationData = None
    return trainData, testData, validationData


# Load the dataset
def readMNISTData(path):
    # f = gzip.open(path, 'rb')
    # train_set, valid_set, test_set = cPickle.load(path)
    # f.close()
    data_file = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(data_file, encoding='latin1')
    data_file.close()

    return train_set, valid_set, test_set
