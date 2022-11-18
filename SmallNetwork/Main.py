from Network import *
import pandas as pd

data = pd.read_csv("data.csv")
trainInputData = data.iloc[:700,:3]
trainOutputData = data.iloc[:700,3:]

network = HardCodeNN(trainInputData, trainOutputData)
print(network.layer1)
print(network.layer2)

print(network.giveOutput(np.array([[40,30,20], [20,30,40]])))




