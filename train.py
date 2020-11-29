from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = [i for i in os.listdir(image_path)]
    image_name_arr = [os.path.join(image_path, i) for i in image_name_arr]
    image_name_arr2 = [i for i in os.listdir(mask_path)]
    image_name_arr2 = [os.path.join(mask_path, i) for i in image_name_arr2]
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        #print(item)
        img = io.imread(item,as_gray = image_as_gray)
        img = cv2.resize(img, (320, 320))
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        item2 = image_name_arr2[index]
        mask = io.imread(item2,as_gray = mask_as_gray)
        mask = cv2.resize(mask, (320, 320))
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        # print(img)
        # cv2.imshow("sample",img)
        # cv2.imshow("samplemask",mask)
        # cv2.waitKey(0)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

#
new_model = keras.models.load_model('saved_model.h5')
new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
imgs_train,imgs_mask_train = geneTrainNpy("Train/X/","Train/Y/")
#new_model.fit(imgs_train, imgs_mask_train, batch_size=5, nb_epoch=2, verbose=1,validation_split=0.1, shuffle=True)
#new_model.save('new_model.h5')
