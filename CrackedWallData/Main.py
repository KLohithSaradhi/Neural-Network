import numpy as np
from src import *

#%%

data = np.load("imagesList.npy", allow_pickle=True)
data.shape

#%%
x_p = np.array(data[20000:, :2500])
x_n = np.array(data[:20000, :2500])

x_p = x_p/255
x_n = x_n/255

y_p = data[20000:, 2500:]
y_p = y_p.flatten()

y_n = data[:20000, 2500:]
y_n = y_n.flatten()

y_lab = np.concatenate((y_p,y_n))

#%%

print(x_n[0].shape)

#%%

y=np.zeros((len(x_p) + len(x_n),2))
for i in range(len(x_p) + len(x_n)):
    y[i,int(y_lab[i])]=1
xtr = np.concatenate((x_p[:15000],x_n[:15000]))
ytr = np.concatenate((y[:15000],y[20000:35000]))
xtst = np.concatenate((x_p[15000:],x_n[15000:]))
ytst = np.concatenate((y[15000:20000],y[35000:40000]))
ytst_lab = np.concatenate((y_lab[15000:20000],y_lab[35000:40000]))

#%%
net=nn(xtr,ytr)
pred=net.predict(xtst)
pred_lab = np.argmax(pred,axis=1)
print("\n numpy neural network with one hidden layer")
mean_error, rmse = accuracy(ytst_lab, pred_lab)

print("mean error:",mean_error)
print("rmse:",rmse)

i=10


while(rmse > 0.3):
    net.train()
    print(i)
    i+=1
    if (i % 10 == 0):
        pred=net.predict(xtst)
        pred_lab = np.argmax(pred,axis=1)
        print("\n numpy neural network with one hidden layer")
        
        mean_error, rmse = accuracy(ytst_lab, pred_lab)
        
        print("mean error:",mean_error)
        print("rmse:",rmse)






#%%

#======using sklearn linear regression======
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xtr, ytr)
pred = reg.predict(xtst)
pred_lab2 = np.argmax(pred,axis=1)
print("\nscikit learn linear regression")
accuracy(ytst_lab,pred_lab2)