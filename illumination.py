    
# Detection and removal based on study [1].
# Note that notations (r_, m_, s_) are adapted from the paper.
import cv2
import numpy as np
import specularity as spc  

image = cv2.imread("S1008L02.jpg",0)
mask = np.ones(image.shape)
for i in range(100,len(image)-90):
    for j in range(80, len(image[0])-80):
        if image[i][j]>230:
            mask[i][j] = 0

img = np.zeros(image.shape)
img = image*mask
for i in range(len(image)):
    for j in range(len(image[0])):
        if mask[i][j] == 0:
            sum_ = 0
            for k in range(-5, 5):
                for l in range(-5, 5):
                    sum_ += image[i + k][j + l]
            sum_ = sum_/300
            for k in range(-5, 5):
                for l in range(-5, 5):
                    img[i + k][j + l] = sum_


# img = mask*image
cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.imwrite("mask.jpg",img)