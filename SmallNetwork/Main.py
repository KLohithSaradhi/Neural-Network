from Network import *
import pandas as pd

data = pd.read_csv("normalizedData.csv")
trainInputData = np.array(data.iloc[:20,1:4])
trainOutputData = np.array(data.iloc[:20,4:])

network = HardCodeNN(trainInputData, trainOutputData)

print(network.layer1)
print(network.layer2)
print("=============")

output = network.giveOutput()

delta = network.backPropagate(output)

network.updateWeights(delta)