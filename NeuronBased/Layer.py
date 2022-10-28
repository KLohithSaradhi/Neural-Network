import numpy as np

def sigmoid(x):
    return 6/(1+np.exp(-1*x))

class Layer:
    
    def __init__(self, noOfWeights, noOfNeurons):
        self.noOfWeights = noOfWeights
        self.noOfNeurons = noOfNeurons
        self.weights = np.random.random((noOfWeights + 1,noOfNeurons))
        
    def giveOutput(self, inputs):
        inputs = np.append(inputs, np.ones((len(inputs), 1)), axis=1)
        self.out = inputs @ self.weights
        self.activatedOutput = sigmoid(self.out)
        return self.activatedOutput
    
#%%
''' 
    
x = Layer(3,2)
IN = [[1,2,3],
[4,5,6],
[7,8,9],
[0,1,2]]

IN = np.array(IN)

x.giveOutput(IN)
'''