import numpy as np

FEATURES_PER_DATAPOINT = 3
NEURONS_IN_LAYER1 = 2
NEURONS_IN_LAYER2 = 1



#brute coding the neural network
class HardCodeNN:
    def __init__(self, inputData, outputData):
        self.inputData = inputData
        self.outputData = outputData
        
        self.layer1 = np.random.random((NEURONS_IN_LAYER1, FEATURES_PER_DATAPOINT)).T
        self.layer2 = np.random.random((NEURONS_IN_LAYER2, NEURONS_IN_LAYER1)).T
        
    def giveOutput(self, inputDatapoint = []):
        if (inputDatapoint != []):
            layer1Output = inputDatapoint @ self.layer1
            layer2Output = layer1Output @ self.layer2
            return (layer1Output, layer2Output)
        
        self.layer1Output = self.inputData @ self.layer1
        self.layer2Output = self.inputData @ self.layer2
    
                    
        