# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras import optimizers
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
from copy import copy
import matplotlib.pyplot as plt

import os
import pandas as pd
from scipy.misc import imread
from glob import glob
import numpy as np
import re 

        
path = "C:/Users/Admin/Desktop/DEV_M2SID_SHOT"
# il faut installer un package pour redimmensionner les images : !pip install python-resize-image
CATEGORY_REPOSITORY = 'C:/Users/Mehdi/Desktop/M2 2017-2018/S1/Analyse et exploitation de données/TER Challenge/DEV_M2SID_VIDEO/'
CATEGORIES = {
    '1001': 'autos_and_vehicles',
    '1009': 'food_and_drink',
    '1011': 'health',
    '1013': 'movies_and_television',
    '1014': 'music_and_entertainment',
    '1016': 'politics',
    '1017': 'religion',
    '1019': 'sports'
}

def table_file_category():
    file_category = []
    for category in os.listdir(CATEGORY_REPOSITORY):
        for file in os.listdir(CATEGORY_REPOSITORY + category + '/'):
            file_catg = {
                'filename': file[0:len(file)-8],
                'category_id': int(category),
                'category_name': CATEGORIES[category]
            }
            file_category.append(file_catg)
    df_category = pd.DataFrame(file_category)
    return df_category
    
def recup_path_img(path):
    ldir = os.listdir(path)
    path_im = []
    path_tmp = {}
    for i in range(len(ldir)):
        if ldir[i][-3:] == 'xml':
            y = "ficxml"
        else:
            y = glob(os.path.join(path+"\\"+ldir[i], '*.jpg'))
            for j in range(len(y)):
                path_tmp = {"filename":ldir[i], "path_img": y[j]}
                path_im.append(path_tmp)
            table_img = pd.DataFrame(path_im)
    return table_img
    
def mergeList(list1,list2, column):
    newlist = pd.merge(list1, list2, on=column)
    return newlist
        
def getCategory(nameImg,listImg):
    for i in range(len(listImg)):
        if listImg['name_img'][i] == nameImg:
            res = listImg['category_id'][i]
            return res                

############### Traitement ###############
table_img = recup_path_img(path)    
t = table_file_category()
nl = mergeList(table_img,t,'filename')

path1 = "C:/Users/Mehdi/Desktop/M2 2017-2018/S1/Analyse et exploitation de données/TER Challenge/extraction/IMAGE/DEV_M2SID_SHOT"
path2 = "C:/Users/Mehdi/Desktop/M2 2017-2018/S1/Analyse et exploitation de données/TER Challenge/extraction/IMAGE/IMG_TRT"

# TODO : faire demande a l'utilisateur pour créer les images
for n_img in nl['path_img']:
    im = Image.open(n_img)
    im = im.resize((100,100))
    niv_gris = im.convert('L')
    nom_image = ""
    nom_image = re.sub(r".*?(?P<ni>\d+\.jpg)", r"\g<ni>", n_img)
    niv_gris.save(path2+"/"+nom_image)

name = []
for i in range(len(nl)):
    name.append(re.sub(r".*?(?P<ni>\d+\.jpg)", r"\g<ni>", nl['path_img'][i]))

nl = nl.assign(name_img = name)    

imglist = os.listdir(path2)
label = []
for elm in imglist:
    label.append(getCategory(elm,nl))
#im1 = np.array([np.array(Image.open(path2+"/"+imglist[0]))])
#m,n=im1.shape[0:2]
immatrix = np.array([np.array(Image.open(path2+"/"+im2)) for im2 in imglist],'f')

data,labels = shuffle(immatrix,label,random_state = 2)
train_data = [data,labels]
X,y = (train_data[0],train_data[1])

################### neural network - CNN ###################

batch_size = 32
nb_classes = len(CATEGORIES)
nb_epoch = 30
img_channel = 1
#nb_filters = 32
#nb_pool = 2
#nb_conv = 3

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
X_train /= 255
X_test /= 255
listcat = [1001, 1009, 1011, 1013, 1014, 1016, 1017, 1019]
Y_train = np_utils.to_categorical([listcat.index(x) for x in Y_train], nb_classes)  # convert to binary matrix
Y_test = np_utils.to_categorical([listcat.index(x) for x in Y_test], nb_classes)

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 100, 100)
    X_test = X_test.reshape(X_test.shape[0], 1, 100, 100)
    input_shape = (1, 100, 100)
else:
    X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    X_test = X_test.reshape(X_test.shape[0], 100, 100, 1)
    input_shape = (100, 100, 1)

### initialisation du modèle classique

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#♥model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))


opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)


'''
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 100, 28)
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
'''























######### test ##########

# opencv => detection de visage ( yolo image science )


### faire de la fusion tardive