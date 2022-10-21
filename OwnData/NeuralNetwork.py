import numpy as np

def sigmoid(array):
    return 1/(1+np.exp(-1 * array))

def prep(array):
    return np.append(array, np.ones((array.shape[0],1)), axis = 1)

class nn:
    def __init__(self, ip, op):
        self.ip = ip
        self.op = op
        
        self.l1_size = 2
        
        self.w1 = np.random.rand(self.l1_size, self.ip.shape[1] + 1).T
        self.w2 = np.random.rand(self.op.shape[1], self.l1_size + 1).T
        
        print(self.w1.shape, self.w2.shape) 
    def train(self):
        
        self.ip = prep(self.ip)
        self.y1 = self.ip @ self.w1
        self.y1_act = sigmoid(self.y1)
        self.y1 = prep(self.y1)
        
        self.y1_act = prep(self.y1_act)
        self.y2 = self.y1_act @ self.w2
        self.y2_act = sigmoid(self.y2)
        
        '''
        self.d_w2 = self.y1_act.T @ ((self.y2_act - self.op) * self.y2 * (1-self.y2))
        self.d_w1 = self.ip.T  @ ((self.y2_act - self.op) * self.y2 * (1-self.y2) @ self.w2.T * self.y1 * (1 - self.y1))
        '''
        
        print(" ip : ", self.ip.shape)
        print(" y1 : ", self.y1.shape)
        print(" y1_act : ", self.y1_act.shape)
        print(" y2 : ", self.y2.shape)
        print(" y2_act : ", self.y2_act.shape)
        