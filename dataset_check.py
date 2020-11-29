import os
from shutil import copyfile
import glob
import cv2

basic = "Train"
a1 = os.path.join(basic, "X")
a2 = os.path.join(basic, "Y")

x = []
y = []
for f in os.listdir(a1):
    x.append(f)

for f in os.listdir(a2):
    y.append(f)
    
x = sorted(x)
y = sorted(y)

ind = 0
for i in x:
    print(i[:-4], y[ind][:-4])
    if i[:-4] == y[ind][:-4]:
        print("yeaa", i)
    else:
        print("noooo", i)
    ind += 1