from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import skimage.io as io

new_model = keras.models.load_model('saved_model.h5')
new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
# new_model.summary()
image = cv2.imread("mask.jpg",0)
dim = image.shape
new_image = image/255.0
inp = new_image
inp = cv2.resize(inp,(320,320))    
image = cv2.resize(image,(320,320))
input_tensor = np.reshape(inp,[1, 320,320,1])
#img = io.imread("S1001L01.jpg",as_gray = True)
#img = cv2.resize(img, (320, 320))
#img = np.reshape(img,img.shape + (1,))
#img=img/255
#input_tensor = img        
output_tensor = new_model.predict(input_tensor)
output = np.reshape(output_tensor,[320,320])
output = cv2.resize(output, (dim[1], dim[0]))
output = output*255
old_image = cv2.imread("mask.jpg",0)
for i in range(len(output)):
    for j in range(len(output[i])):
        if output[i][j] > 128:
            output[i][j] = old_image[i][j]
        else:
            output[i][j] = 0
# output = cv2.resize(output, (dim[1], dim[0]))
# old_image = cv2.imread("img3.jpg",0)
# for i in range(len(output)):
#     for j in range(len(output[0])):
#         if output[i][j]>0:
#             output[i][j] = int(old_image[i][j])
        # print(output[i][j])
# output = np.array(output,dtype=uint8)
# print(final_image)
# print("hi")
# cv2.imshow("old",old_image)
cv2.imshow("out.jpg",output)
cv2.imwrite("out.jpg",output)
cv2.waitKey(0)
