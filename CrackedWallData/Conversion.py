#from PIL import Image
import cv2
import os
img = cv2.imread(os.path.join('./','negative/00001.jpg'), 0)
path = './'
cv2.imwrite(os.path.join(path , 'negative/00001.jpg'), img)


#%%
from math import *

convert = lambda i : '0'*(5 - len(str(i))) + str(i)

#%%
import cv2
import os


for i in range(1,20001):
    
    
    try:
        path = 'positive/' + convert(i) + '.jpg'
        img = cv2.imread(os.path.join('./', path))
        img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join('./' , path), img)
        print("done : ",i)
    except:
        path = 'positive/' + convert(i) + '_1.jpg'
        img = cv2.imread(os.path.join('./', path))
        img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join('./' , path), img)
        print("done : ",i)
        

#%%
from PIL import Image
import numpy as np
import os

imagesList = []

for i in range(1,20001):
    
    try:
        path = 'negative/' + convert(i) + '.jpg'
        
        img = Image.open(os.path.join('./', path))
        imagesList += [list(np.asarray(img).flatten()) + [0]]
        print("done : ",i)
    except:
        path = 'negative/' + convert(i) + '_1.jpg'
        
        img = Image.open(os.path.join('./', path))
        imagesList += [list(np.asarray(img).flatten()) + [0]]
        print("done : ",i)
        
for i in range(1,20001):
    try:
        path = 'positive/' + convert(i) + '.jpg'
        
        img = Image.open(os.path.join('./', path))
        imagesList += [list(np.asarray(img).flatten()) + [1]]
        print("done : ",i)
    except:
        path = 'positive/' + convert(i) + '_1.jpg'
        
        img = Image.open(os.path.join('./', path))
        imagesList += [list(np.asarray(img).flatten()) + [1]]
        print("done : ",i)
#%%

imagesList = np.array(imagesList)
np.save("imagesList",imagesList)
    
#%%

x = np.load('imagesList.npy', allow_pickle=True)


