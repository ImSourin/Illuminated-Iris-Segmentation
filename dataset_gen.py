import os
from shutil import copyfile
import glob
import cv2

basic = "CASIA-Iris-Interval"
arr = [os.path.join(basic, os.path.relpath(name)) for name in os.listdir(basic)]
arr2 = [os.path.join(i, "L") for i in arr]
arr3 = [os.path.join(i, "R") for i in arr]
arr = arr2
for x in arr3:
    arr.append(x)

yy = []
zz = []
for x in arr:
    for filename in os.listdir(x):
        if filename[-6:] == "01.jpg":
            yy.append(os.path.join(x, filename))
            copyfile(os.path.join(x, filename), os.path.join("train", filename))

#images = []
#for filename in arr:
#    
##    images.append(os.join(y, ))