import numpy as np

FEATURES_PER_DATAPOINT = 3
NEURONS_IN_LAYER1 = 2
NEURONS_IN_LAYER2 = 1

def noActivation(x):
    return x
def noActivationDerivative(x):
    return np.ones(x.shape)

#brute coding the neural network
class HardCodeNN:
    def __init__(self, inputData, outputData):
        self.inputData = inputData
        self.outputData = outputData
        
        self.noOfRecords = self.inputData.shape[0]
        self.noOfFeatures = self.inputData.shape[1]
        
        self.layer1 = np.random.random((NEURONS_IN_LAYER1, FEATURES_PER_DATAPOINT)).T
        self.layer2 = np.random.random((NEURONS_IN_LAYER2, NEURONS_IN_LAYER1)).T
        
    def giveOutput(self, inputDatapoint = []):
        if (inputDatapoint != []):
            layer1Output = inputDatapoint @ self.layer1
            layer2Output = layer1Output @ self.layer2
            return (layer1Output, layer2Output)
        
        self.layer1Output = self.inputData @ self.layer1
        self.layer2Output = self.layer1Output @ self.layer2
        
        return self.layer2Output
    
    def backPropagate(self, predicted):
        differenceInOutput = (self.outputData - predicted)     
        self.error = (self.outputData - predicted)**2
        self.errorMeasure = sum(np.reshape(self.error, -1)) / len(self.outputData)
        
        self.dw2 = self.layer1Output.T @ (differenceInOutput * noActivationDerivative(predicted))
        print(self.dw2)
        
        temp = (differenceInOutput * noActivationDerivative(predicted)) * (noActivationDerivative(self.layer1Output) @ self.layer2)
        self.dw1 = self.inputData.T @ temp
        print(self.dw1)
        
        return (self.dw1, self.dw2)
    
    def updateWeights(self, delta):
        self.layer1 -= delta[0]
        self.layer2 -= delta[1]
        
        
                             
        