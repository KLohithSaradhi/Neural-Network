import cv2
import numpy as np
#%%

img = cv2.imread("./HOUSE_TEST/001.jpeg")

img = cv2.resize(img, (50,50), interpolation = cv2.INTER_AREA)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



ip = np.asarray(img).flatten()
ip = ip/255
ip = np.concatenate((ip, [1]))

w1 = np.load("layer1.npy")
w2 = np.load("layer2.npy")
#%%

l1 = ip @ w1

l1 = np.concatenate((l1,[1]))

l2 = l1 @ w2