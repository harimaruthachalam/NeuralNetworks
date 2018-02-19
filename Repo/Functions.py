import numpy as np
# import Repo.SoftMax as SoftMax
# import Repo.CommonUtilityFunctions.accuracy
# from Repo.CommonUtilityFunctions import accuracy
# from src.Utility.CommonUtilityFunctions import accuracy
# import Repo.CommonUtilityFunctions as cuf


def ValidateDimensionsWithBiasAndOutput(data,weights,biases,output):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1 and output is Dim(L)*NoOfExaples
    if (data.shape[0] != weights.shape[1] or biases.shape[0] != weights.shape[0] or output.shape[0]!=weights.shape[0] or output.shape[1]!=data.shape[1] ):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape, 'biases:', biases.shape, 'output:', output.shape)
        raise ValueError('Incorrect dimmension given to gradient function')


def ValidateDimensionsWithOutput(data,weights,output):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1) and output is Dim(L)*NoOfExaples
    if (data.shape[0] != weights.shape[1]  or output.shape[0]!=weights.shape[0] or output.shape[1]!=data.shape[1] ):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape,  'output:', output.shape)
        raise ValueError('Incorrect dimmension given to gradient function')


def ValidateDimensionsWithBias(data,weights,biases):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1
    if (data.shape[0] != weights.shape[1] or biases.shape[0] != weights.shape[0]):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape, 'biases:', biases.shape)
        raise ValueError('Incorrect dimmension given to sigmoid function')


def   ValidateDimensionsWithActivationAndGradients(activation,biases):
    #todo: Complete Function
    pass


def ValidateDimensions(data, weights):
    # Validates data to be Dim(L-1)*NoOfExaples, Weights to be Dim(L)*Dim(L-1), Biases to be Dim(L)*1
    if (data.shape[0] != weights.shape[1]):
        print('Error, please check the domension of input matrices')
        print('data:', data.shape, 'weights', weights.shape)
        raise ValueError('Incorrect dimmension given to sigmoid function')


def IntergrateBiasWithWeightsAndData(data, weights, biases):
    # Input: data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    # Output: data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)*Dim(L-1)+1
    data = IntergrateBiasAndData(data)
    weights = IntergrateBiasWithWeights(weights,biases)
    return data, weights

def IntergrateBiasAndData(data):
    # Input: data is Dim(L-1)*NoOfExaples
    # Output: data is Dim(L-1)+1(bias)*NoOfExaples
    data = np.append(data, np.ones((1, data.shape[1])), axis=0)
    return data

def IntergrateBiasWithWeights(weights,biases):

    weights = np.append(weights, biases, axis=1)
    return weights

def DisIntergrateBiasFromWeights(weights,biasRequired=False):
    # Output: Weights is Dim(L)*Dim(L-1)+1
    # Input: Weights is Dim(L)*Dim(L-1), Biases is Dim(L-1)*1
    weightsWithoutBias=weights[:,0:weights.shape[1]-1]
    biases=weights[:,weights.shape[1]-1]
    biases=np.transpose(np.array([biases]))
    if biasRequired:
        return weightsWithoutBias,biases
    else:
        return weightsWithoutBias


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 0) != np.argmax(labels, 0))
            / predictions.shape[1])

def CrossEntropyWithSoftMaxAndBias(data,weights,biases,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    outputActivations=SoftMaxWithBias(data,weights,biases)
    return -np.mean(np.einsum('ij,ji->i', np.transpose(targetOutput), np.log(outputActivations)))

def CrossEntropyWithSoftMax(data,weights,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    outputActivations=SoftMaxWithBias(data,weights)
    return -np.mean(np.einsum('ij,ji->i', np.transpose(targetOutput), np.log(outputActivations)))

def CrossEntropy(outputActivations,targetOutput):
    #todo : Add a dimension check and set dimension check to false in softmax call and add comment
    return -np.mean(np.einsum('ij,ji->i', np.transpose(targetOutput), np.log(outputActivations)))
    #-np.mean(np.diag(np.matmul(np.transpose(targetOutput),np.log(outputActivations))))




def CrossEntropyWithSoftMaxGradients(outputActivations,targetOutput):
    #targetOutput=#noOfClasses*#noOfExamples outputActivafrom OutputFunctions import SoftMaxtion=#noOfClasses*#noOfExamples
    #todo : Add a dimension check and set dimension check to false in softmax call
    return -(targetOutput-outputActivations)




def WriteLog(net, trainData, trainTragets, step, epoch, lr, valData=None, valTargets=None, testData=None
             , testTargets=None):
    output , _ =net.FeedForward(trainData)
    loss=net.LossFunction[net.lossFunctionName](output, trainTragets)
    eer = accuracy(output, trainTragets)
    filename= net .logDir + '/log_loss_train.txt'
    WriteLossLog(epoch, step, loss, lr, filename)
    filename = net.logDir + '/log_err_train.txt'
    WriteEERLog(epoch, step, eer, lr, filename)

    # if valData!=None :
    #     output=FeedForwadData(valData,net)
    #     loss = net.LossFunction[net.lossFunctionName](output, valTargets)
    #     eer = accuracy(output, valTargets)
    #     filename = net.logDir + '/log_loss_valid.txt'
    #     WriteLossLog(epoch, step, loss, lr, filename)
    #     filename = net.logDir + '/log_err_valid.txt'
    #     WriteEERLog(epoch, step, eer, lr, filename)
    #
    # if testData!=None :
    #     output = FeedForwadData(testData, net)
    #     loss = net.LossFunction[net.lossFunctionName](output, testTargets)
    #     eer = accuracy(output, testTargets)
    #     filename = net.logDir + '/log_loss_test.txt'
    #     WriteLossLog(epoch, step, loss, lr, filename)
    #     filename = net.logDir + '/log_err_test.txt'
    #     WriteEERLog(epoch, step, eer, lr, filename)

def WriteLossLog(epoch ,step, loss, lr, filename):
    text_file = open(filename, "a+")
    text_file.write("Epoch %s, Step %s, Loss: %f, lr: %f \n" % (epoch, step,loss, lr))
    text_file.close()

def WriteEERLog(epoch, step, eer, lr, filename):
    text_file = open(filename, "a+")
    text_file.write("Epoch %s, Step %s, Error: %f, lr: %f \n"  % ( epoch, step, eer, lr))
    text_file.close()

def FeedForwadData(data,net):
    batchSize=5000
    if data.shape[1] > 5000:
        i = 0
        posteriors = []
        while i + 5000 <= data.shape[1]:
            output, _ = net.FeedForward(data[:, i:i + 5000])
            posteriors.append(output)
            i = i + 5000
        output, _ = net.FeedForward(data[:, i:])
        posteriors.append(output)
        output = np.concatenate(posteriors, axis=1)

    else:
        output, _ = net.FeedForward(data)
    return output


def PureLin(data, weights):
    preActivation=np.matmul(weights,data)
    return preActivation


def PureLinWithBias(data,weights,biases,validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithBias(data,weights,biases)
    DataWithBias,WeightsWithBias=cuf.IntergrateBiasWithWeightsAndData(data,weights,biases)
    return PureLin(DataWithBias,WeightsWithBias)

def PureLinGradients(activation, gradients, validationRequired=True):
    #todo:Correct the function
    return gradients


# from Repo.CommonUtilityFunctions import ValidateDimensionsWithBiasAndOutput, ValidateDimensionsWithOutput, \
#     ValidateDimensionsWithBias, ValidateDimensions, IntergrateBiasWithWeightsAndData


def LogSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        ValidateDimensionsWithBias(data, weights, biases)
    data,weights= IntergrateBiasWithWeightsAndData(data, weights, biases)
    return LogSigmoid(data, weights, False)

def LogSigmoid(data, weights, validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data, weights)
    return np.divide(1.0,(1.0+ np.exp(-1*np.matmul(weights,data))))

def TanSigmoidWithBias(data,weights,biases,validationRequired=True):
    # data is Dim(L-1)*NoOfExaples, Weights is Dim(L)*Dim(L-1), Biases is Dim(L)*1
    if validationRequired:
        ValidateDimensionsWithBias(data, weights, biases)
    data,weights= IntergrateBiasWithWeightsAndData(data, weights, biases)
    return TanSigmoid(data, weights, False)

def TanSigmoid(data,weights,validationRequired=True):
    # data is Dim(L-1)+1(bias)*NoOfExaples, Weights is Dim(L)+1*Dim(L-1)
    if validationRequired:
        ValidateDimensions(data, weights)
    preActivation=np.matmul(weights,data)
    return np.divide(np.exp(preActivation)-np.exp(-1*preActivation),np.exp(preActivation)+np.exp(-1*preActivation))

def ReLU(data,weights,validationRequired=True):
    if validationRequired:
        ValidateDimensions(data, weights)
    preActivation=np.matmul(weights,data)
    return np.maximum(preActivation,0)


#######################################Sigmoid Gradiants###############################################################




def LogSigmoidGradients(activations, gradients, validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithActivationAndGradients(activations, gradients)
    return np.multiply(np.multiply( gradients, activations), (1 - activations))



def TanSigmoidGradients(activations, gradients, validationRequired=True):
    #todo: remove weights
    if validationRequired:
        ValidateDimensionsWithActivationAndGradients(activations, gradients)
    return np.multiply(np.multiply(gradients, (1 + activations)), (1 - activations))

def ReLUGradients(activations, gradients, validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithActivationAndGradients(activations, gradients)
        reluGradient=activations
        reluGradient[np.greater(activations,0)]=1
    return np.multiply(gradients, reluGradient)

#######################################################################################################################


def SoftMax(data, weights):
    preActivation=np.matmul(weights,data)
    return np.divide(np.exp(preActivation),np.sum(np.exp(preActivation), axis=0))


def SoftMaxWithBias(data,weights,biases,validationRequired=True):
    if validationRequired:
        ValidateDimensionsWithBias(data,weights,biases)
    DataWithBias,WeightsWithBias=IntergrateBiasWithWeightsAndData(data,weights,biases)
    return SoftMax(DataWithBias,WeightsWithBias)



def SquaredError(outputActivations,targetOutput):
    #todo : Add a dimension check
    return (1.0/2.0)*np.linalg.norm(outputActivations-targetOutput)
def SquaredErrorGradients(outputActivations,targetOutput):
    #todo : Add a dimension check
    return outputActivations-targetOutput

