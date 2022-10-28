from Layer import *

def showWeights(layer):
    for i in layer:
        print(i.weights)


INPUT = np.array([[1, 2, 3],
         [4, 5, 3],
         [2, 4, 6],
         [2, 0, 4],
         [0, 2, 0]])

OUTPUT = np.array([[2.5],
                   [4],
                   [5],
                   [2],
                   [1]])

layer1 = Layer(3,2)
layer2 = Layer(2,1)


for i in range(100):
    out1 = layer1.giveOutput(INPUT)
    out2 = layer2.giveOutput(out1)
    
    error = (out2 - OUTPUT)**2
    
    dw2 = (out1.T @ error)/100
    out1_withdummy = np.append(out1, np.ones((len(out1), 1)), axis=1)
    
    segment1 =  out1_withdummy * (1 - out1_withdummy)
    dw1 = (INPUT.T @ (error @ layer2.weights.T * segment1))/100
    
    layer1.weights -= dw1
    layer2.weights -= dw2
