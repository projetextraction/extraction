
# lib et biblio à importer
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import os
import pandas as pd
from scipy.misc import imread, imresize
from glob import glob

# chemin du dossier contenant les images clés des vidéos        
path = "IMAGE/DEV_M2SID_SHOT"

# le dossier "DEV_M2SID_VIDEO" n'est pas disponible sur le git, il a été fourni en début de projet
CATEGORY_REPOSITORY = 'C:/Users/Mehdi/Desktop/M2 2017-2018/S1/Analyse et exploitation de données/TER Challenge/DEV_M2SID_VIDEO/'

# dictionnaire contenant toutes les categories
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

# fonction permetant de recuperer la categorie d'une video
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

# fonction permettant de recuperer le chemin et le nom d'une image
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
   
# fonction permettant de fusionner deux listes en fonction d'une colonne en commun        
def mergeList(list1,list2, column):
    newlist = pd.merge(list1, list2, on=column)
    return newlist
 
# fonction permettant de recuperer la categorie d'une image        
def getCategory(nameImg,listImg):
    for i in range(len(listImg)):
        if listImg['name_img'][i] == nameImg:
            res = listImg['category_id'][i]
            return res    

# constitution des tableaux contenant les informations sur les images
table_img = recup_path_img(path)    
t = table_file_category()
nl = mergeList(table_img,t,'filename')

liste_video = nl['filename'].drop_duplicates(keep="first")

# traitements des images
data_img = []
for elm in liste_video:
    liste_image = []
    liste_image = os.listdir(path+'/'+elm)
    concat_img = []
    for im in liste_image:
            img = imread(path+'/'+elm+'/'+im)
            img = imresize(img,(100,100))
            concat_img.append(img)
    data_img.append(concat_img)

# Calcul de l'image moyenne pour chaque vidéo et stockage dans la liste "data_img_concat"
data_img_concat = []
for i in range(len(data_img)):
    concat_bool = True
    concat = []

    if (len(data_img[i]) < 2):
        data_img_concat.append(data_img[i][0])     
    else:
        for elm in data_img[i]:
            if concat_bool == True:
                concat = elm
                concat_bool = False
            else:
                concat = (concat+elm)/2
        data_img_concat.append(concat)     

# on associe une categorie à nos images        
label = []
for elm in liste_video.index:
    label.append(nl.iloc[elm]['category_id'])

# on melange aléatoirement les informations pour construire notre jeu de données
data,labels = shuffle(data_img_concat,label,random_state = 2)
train_data = [data,labels]
X,y = (train_data[0],train_data[1])

    
################### neural network - CNN ###################

# paramètres du modèle
batch_size = 32
nb_classes = len(CATEGORIES)
nb_epoch = 20
img_channel = 3

# création du jeu de test et du jeu d'apprentissage 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33, random_state = 4)
X_train /= 255
X_test /= 255
listcat = [1001, 1009, 1011, 1013, 1014, 1016, 1017, 1019]
Y_train = np_utils.to_categorical([listcat.index(x) for x in Y_train], nb_classes)   # convertion en matrice binaire (1 pour la bonne classe)
Y_test = np_utils.to_categorical([listcat.index(x) for x in Y_test], nb_classes)

# Determine notre format d'entrée du modèle
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], img_channel, 100, 100)
    X_test = X_test.reshape(X_test.shape[0], img_channel, 100, 100)
    input_shape = (img_channel, 100, 100)
else:
    X_train = X_train.reshape(X_train.shape[0], 100, 100, img_channel)
    X_test = X_test.reshape(X_test.shape[0], 100, 100, img_channel)
    input_shape = (100, 100, img_channel)

### initialisation du modèle CNN

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=input_shape)) # couche d'entrée du modèle
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten()) # transforme notre matrice en vecteur
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax')) # couche contenant un neurone par classe


opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)


score = model.evaluate(X_test, Y_test, verbose=0)
            