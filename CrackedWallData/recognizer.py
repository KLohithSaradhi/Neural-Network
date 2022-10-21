from scipy.io import loadmat
import numpy as np

class nn:
    def __init__(self,ip,op):
        self.ip=ip
        self.op=op
        print(ip.shape,op.shape)
        self.l1_size=500
        self.l2_size=50
        self.w1=np.random.rand(self.l1_size,self.ip.shape[1]+1).T*0.001
        self.w2=np.random.rand(self.l2_size,self.l1_size+1).T*0.001
        self.w3=np.random.rand(self.op.shape[1],self.l2_size+1).T*0.001
        
        print(self.w1.shape,self.w2.shape)
        for i in range(10):
            print(i)
            self.train()
    
    def sig(self,x):
        return 1/(1+np.exp(-x))
    
    def train(self):
        #===front propogation===#
        ip=np.append(np.ones((len(self.ip),1)),self.ip,axis=1)
        l1=self.sig(ip@self.w1)
        l1=np.append(np.ones((len(l1),1)),l1,axis=1)
        print(l1.shape)
        l2=self.sig(l1@self.w2)
        l2=np.append(np.ones((len(l2),1)),l1,axis=1)
        pred=self.sig(l2@self.w3)
        
        
        #===back propogation===#
        #error at each layer
        d3 = (pred - self.op)
        d2 = (d3 @ self.w3.T * l2 * (1-l2))
        d1 = (d2 @ self.w2.T * l1 *(1-l1))
        d2 = d2[:,1:]
        d1 = d1[:,1:]
        #altering weights
        self.w3-= (l2.T @ d3)/20000
        self.w2-= (l1.T @ d2)/20000
        self.w1-= (ip.T @ d1)/20000
        
    def predict(self,ip):
        ip=np.append(np.ones((len(ip),1)),ip,axis=1)
        l1=self.sig(ip@self.w1)
        l1=np.append(np.ones((len(l1),1)),l1,axis=1)
        pred=self.sig(l1@self.w2)
        return pred

def accuracy(ytst,ypred):
    return (np.mean(np.abs((ypred-ytst).astype(bool))), np.sqrt(((ypred-ytst).astype(bool)**2).mean()))
