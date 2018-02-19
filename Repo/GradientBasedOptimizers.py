import numpy as np
import pickle
import Repo.Functions as fns
import Repo.Network as Network
import matplotlib.pyplot as plot


def BatchGradientDecent(net, trainData, trainTargets, eta, itr, valData=None, valTargets=None, testData=None, testTargets=None,annel=False):
    eta, _ = SetInitialETA(net, trainData, trainTargets, eta)
    lossToPlotTrain = []
    lossToPlotVal = []
    for i in range(0,itr):
        networkOutput,layerOutputs = net.FeedForward(trainData)
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, trainTargets))
        gradients = net.BackProbGradients(trainTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers+1):
            net.weights[j]= net.weights[j] - (eta / trainData.shape[1]) * gradients[j]
        plot.close('all')
        # lossToPlotTrain.append(CrossEntropy.CrossEntropy(networkOutput, trainTargets))
        # valOutput,_ = net.FeedForward(valData)
        # lossToPlotVal.append(CrossEntropy.CrossEntropy(valOutput, valTargets))
        # plot.plot(lossToPlotTrain)
        # plot.plot(lossToPlotVal)
        # plot.legend(['TrainErr', 'ValErr'])
        # plot.show()

        valOutput,_ = net.FeedForward(valData)
        valLoss = fns.CrossEntropy(valOutput, valTargets)
        print('Val Loss: ', valLoss)
        if net.logDir!=None and i%250==0 :
            fns.WriteLog(net, trainData, trainTargets, i, i, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecent(net, trainData, trainTargets,  itr, batchSize, eta=0.5,valData=None, valTargets=None,
                            testData=None, testTargets=None,annel=False,regularization=False,lamda=0.1):
    batchStart=0
    step = 0
    epoch = 0
    aneelCount=0
    previousEpochValLoss=np.inf
    eta, _ = SetInitialETA(net, trainData[:, 0:batchSize], trainTargets[:, 0:batchSize], eta)
    for i in range(0, itr):
        #batchSelection=np.random.choice(np.arange(trainData.shape[1]), batchSize)
        step=step+1
        batchData=trainData[:,batchStart:batchStart+batchSize]
        batchTargets=trainTargets[:,batchStart:batchStart+batchSize]
        batchStart=batchSize+batchStart
        if(batchStart>=trainData.shape[1]):
            epoch=epoch+1
            batchStart= batchStart-trainData.shape[1]
            if annel and valData !=None:
                previousEpochValLoss,tempNet=HandleAneeling(net,valData,valTargets,previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
            step=0
        networkOutput, layerOutputs = net.FeedForward(batchData)
        print('Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            net.weights[j] = net.weights[j] - eta / batchSize * gradients[j]
        if net.logDir != None and step%250==0:
            fns.WriteLog(net, batchData, batchTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net


def MiniBatchGradientDecentWithMomentum(net, trainData, trainTargets, itr, batchSize, eta=0.5, gamma=0.5, valData=None,
                                        valTargets=None,testData=None, testTargets=None,annel=False,
                                        regularization=False,lamda=0.1):
    deltaWeights=[None]* (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    aneelCount = 0
    previousEpochValLoss = np.inf
    eta,gamma=SetInitialETA(net,trainData[:,0:batchSize],trainTargets[:,0:batchSize],eta,gamma)
    for i in range(0, itr):
        step = step + 1
        batchData = trainData[:, batchStart:batchStart + batchSize]
        batchTargets = trainTargets[:, batchStart:batchStart + batchSize]
        batchStart = batchSize + batchStart
        networkOutput, layerOutputs = net.FeedForward(batchData)
        if (batchStart >= trainData.shape[1]):
            epoch = epoch + 1
            batchStart = batchStart - trainData.shape[1]
            step = 0
            if annel and valData !=None:
                previousEpochValLoss, tempNet = HandleAneeling(net, valData, valTargets, previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount >3:
                        return net
        print('Mini Batch Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            if deltaWeights[j] == None:
                deltaWeights[j]= eta / batchSize * gradients[j]
            else:
                deltaWeights[j] = eta / batchSize * gradients[j] + gamma *deltaWeights[j]
            net.weights[j] = net.weights[j] - deltaWeights[j]
        if net.logDir != None and step%250==0:
            fns.WriteLog(net, batchData, batchTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def NestrovAccelaratedGradientDecent(net, trainData, trainTargets, itr, batchSize, eta=0.5, gamma=0.5, valData=None,
                                        valTargets=None,testData=None, testTargets=None,regularization=False,lamda=0.1,
                                     annel=False):
    deltaWeights=[None]* (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    previousEpochValLoss = np.inf
    aneelCount = 0
    eta, gamma = SetInitialETA(net, trainData[:, 0:batchSize], trainTargets[:, 0:batchSize], eta, gamma)
    for i in range(0, itr):
        step = step + 1
        batchData = trainData[:, batchStart:batchStart + batchSize]
        batchTargets = trainTargets[:, batchStart:batchStart + batchSize]
        batchStart = batchSize + batchStart
        networkOutput, layerOutputs = net.FeedForward(batchData)
        if (batchStart >= trainData.shape[1]):
            epoch = epoch + 1
            batchStart = batchStart - trainData.shape[1]
            step = 0
            if annel and valData !=None:
                previousEpochValLoss, tempNet = HandleAneeling(net, valData, valTargets, previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta*(3.0/4.0)
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
        print('Mini Batch Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        oldWeights=net.weights
        for j in range(0, net.noOfLayers + 1):
            if deltaWeights[j] != None:
                net.weights[j] = net.weights[j] - gamma * deltaWeights[j]
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j]=gradients[j]+lamda* net.weights[j]
            if deltaWeights[j] == None:
                deltaWeights[j]= eta / batchSize * gradients[j]
            else:
                deltaWeights[j] = eta / batchSize * gradients[j] + gamma *deltaWeights[j]
            net.weights[j] = oldWeights[j] - deltaWeights[j]
        if net.logDir != None and step%250==0:
            fns.WriteLog(net, batchData, batchTargets, step, epoch, eta, valData, valTargets, testData, testTargets)
    return net

def AdamOptimizer(net, trainData, trainTargets, itr, batchSize, eta=0.5,b1 = 0.9,b2 = 0.999, valData=None,
                                        valTargets=None,testData=None, testTargets=None,annel=False,
                                        regularization=False,lamda=0.1):
    flag = True
    mt=[None]* (net.noOfLayers + 1)
    vt = [None] * (net.noOfLayers + 1)
    batchStart = 0
    step = 0
    epoch = 0
    aneelCount = 0
    previousEpochValLoss = np.inf
    eta, _ = SetInitialETA(net, trainData[:, 0:batchSize], trainTargets[:, 0:batchSize], eta)

    lossToPlotTrain = []
    lossToPlotVal = []
    for i in range(0, itr):
        step = step + 1
        batchData = trainData[:, batchStart:batchStart + batchSize]
        batchTargets = trainTargets[:, batchStart:batchStart + batchSize]
        batchStart = batchSize + batchStart
        networkOutput, layerOutputs = net.FeedForward(batchData)
        if (batchStart >= trainData.shape[1]):
            epoch = epoch + 1
            batchStart = batchStart - trainData.shape[1]
            step = 0
            if annel and valData !=None:
                previousEpochValLoss, tempNet = HandleAneeling(net, valData, valTargets, previousEpochValLoss)
                if tempNet !=None:
                    net=tempNet
                    eta=eta*3.0/4.0
                    aneelCount += 1
                    if aneelCount > 3:
                        return net
        print('Mini Batch Loss:', net.LossFunction[net.lossFunctionName](networkOutput, batchTargets))
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            if regularization:
                gradients[j] += lamda * net.weights[j]
            if mt[j] is None:
                mt[j]= (1-b1)* gradients[j]
                vt[j]= (1-b2) * np.square(gradients[j])
            else:
                mt[j] = b1*mt[j]+(1 - b1) * gradients[j]
                vt[j] = b2*vt[j]+(1 - b2) * np.square(gradients[j])
            net.weights[j] = net.weights[j] - (eta/batchSize)* np.multiply((1/np.sqrt(vt[j]+1e-8)), gradients[j])
        plot.close('all')
        lossToPlotTrain.append(fns.CrossEntropy(networkOutput, batchTargets))
        valOutput,_ = net.FeedForward(valData)
        valLoss = fns.CrossEntropy(valOutput, valTargets)
        print('Val Loss: ', valLoss)
        if net.logDir != None and step%250==0:
            fns.WriteLog(net, batchData, batchTargets, step, epoch, eta, valData, valTargets, testData, testTargets)

    lossToPlotVal.append()
    plot.plot(lossToPlotTrain)
    plot.plot(lossToPlotVal)
    plot.legend(['TrainErr', 'ValErr'])
    plot.show()
    return net


############################## Pre Training Implementation ############################################################

def PreTrainNetwork(net, trainData, itr, batchSize, eta=0.5, gamma=0.5, valData=None,annel=False,
                                        regularization=False,lamda=0.1,logDir='/tmp'):
    for i in range(0,net.noOfLayers):
        print('Layer ' + str(i) + ' Pretraining')
        preTrainNet=Network(net.hiddenLayers[:i+1],net.activationFunctionNames[:i+1],'PureLin',
                            'SquaredError',trainData.shape[0],trainData.shape[0],logDir);
        for j in range(0,i):
            preTrainNet.weights[j]=net.weights[j]
        preTrainNet=MiniBatchGradientDecentWithMomentum(preTrainNet,trainData,trainData,itr,batchSize,
                                                        eta=eta,gamma=gamma,valData=valData,valTargets=valData,
                                                        annel=annel,regularization=regularization,lamda=lamda)
        for j in range(0, i+1):
            net.weights[j]=preTrainNet.weights[j]
    return net

def HandleAneeling(net,valData,valTargets,previousEpochValLoss):
    valOuput, _ = net.FeedForward(valData)
    presentValLoss = net.LossFunction[net.lossFunctionName](valOuput, valTargets)
    if presentValLoss < previousEpochValLoss:
        with open(net.logDir+"/nnet_temp.pickle", "wb") as output_file:
            pickle.dump(net, output_file)
        return presentValLoss,None
    else:
        with open(net.logDir+"/nnet_temp.pickle", "rb") as input_file:
            net = pickle.load(input_file)
        return previousEpochValLoss,net


def SetInitialETA(net, batchData, batchTargets, eta, gamma=None):
    networkOutput, _ = net.FeedForward(batchData)
    previousLoss=net.LossFunction[net.lossFunctionName](networkOutput, batchTargets)
    previousWeights=net.weights[:];
    while 1:
        networkOutput, layerOutputs = net.FeedForward(batchData)
        gradients = net.BackProbGradients(batchTargets, networkOutput, layerOutputs)
        for j in range(0, net.noOfLayers + 1):
            net.weights[j] = net.weights[j] -  (eta / batchData.shape[1]) * gradients[j]
        networkOutput, _ = net.FeedForward(batchData)
        loss=net.LossFunction[net.lossFunctionName](networkOutput, batchTargets)
        print('loss: ' + str(loss) + ' ' + 'previous Loss' + str(previousLoss))
        if loss<previousLoss:
            break
        else:
            net.weights=previousWeights[:]
            eta=eta*(3.0/4.0)
            if gamma!=None:
                gamma=gamma*(3.0/4.0)
            print('Reducing Learning Rate to' + str(eta))
    return eta,gamma

