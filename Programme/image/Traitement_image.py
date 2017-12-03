# -*- coding: utf-8 -*-




import os
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

'''
#####Traitement d'une image####
#Histogramme de couleurs
def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)
#    
for idx, c in enumerate(colors):
        plt.bar(idx, c[0], color=hexencode(c[1]),edgecolor=hexencode(c[1]))
'''


#Couleur dominante
def calcul_couleur_frequente(img):
    width, height = img.size

    r_total = 0
    g_total = 0
    b_total = 0

    count = 0
    for x in range(0, width):
        for y in range(0, height):
            r, g, b = img.getpixel((x,y))
            r_total += r
            g_total += g
            b_total += b
            count += 1

    return [r_total/count, g_total/count, b_total/count]

#Récuperation chemin des répertoires contenant les vidéos 
path =  os.path.join(os.getcwd(), "Data\\Video\\DEV_M2SID_SHOT")


average_colors = []
for repo in os.listdir(path):
    avgs_colors = []
    if (repo.endswith(".xml")):
        continue
    else:
        for img_file in os.listdir(path + '/' + repo):
            goal_dir = path + '/' + repo + '/' + img_file
            im = Image.open(goal_dir)
            im = im.resize((100,100))
            avgs_colors.append(calcul_couleur_frequente(im))
        mat_avg_colors  = np.asmatrix(avgs_colors)
        mean_avg_colors = mat_avg_colors.sum(axis=0) / mat_avg_colors.shape[0]
        average_colors.append({'filename': repo, 'avg_color': mean_avg_colors.tolist()[0]})
        df_average_colors = pd.DataFrame(average_colors)

df_average_colors.to_csv(os.path.join(os.getcwd(), "Programme\\mis_en_forme\\")+'moy_couleur.csv', sep=';', decimal=',')  


#import d'un fichier csv pour récupérer les atégories associées aux images
path_all_data = os.path.join(os.path.dirname(os.getcwd()), "extraction\\Programme\\mis_en_forme")
data = pd.read_csv(
path_all_data +"\data_cat_lang_tags_text.csv", sep = ";" )

#jointure
data_colors = pd.merge(data , df_average_colors, on='filename')


goal_dir = os.path.join(os.getcwd(), "Programme\\Texte")
os.chdir(goal_dir)

#fonction séparation test - apprentissage
from  fonctions_text import ech_data

#67% train et 33% test
X_train_col, X_test_col, y_train_col, y_test_col = ech_data(data_colors, "avg_color", "category_id", 0.33)   

X_train_col = X_train_col.reset_index(drop=True)
y_train_col = y_train_col.reset_index(drop=True)
X_test_col = X_test_col.reset_index(drop=True)
y_test_col = y_test_col.reset_index(drop=True)

X_train = [X_train_col.iloc[i] for i in range(len(X_train_col))]
X_test = [X_test_col.iloc[i] for i in range(len(X_test_col))]

#random forest classifier
rfc = RandomForestClassifier(n_jobs=-1, oob_score = True) 
 
# Recherche des paramètres optimaux pour le random forest
param_grid = { 
           "n_estimators" : [100, 200, 300, 400, 500],
           "max_depth" : [1, 2,3,4],
           "max_features" : ["auto", "sqrt", "log2", None],
           "min_samples_leaf" : [1, 2, 3, 4 ]}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train_col)

print(CV_rfc.best_params_)

rfc = RandomForestClassifier(n_estimators=
                             300, max_depth=1, max_features=None, min_samples_leaf = 1)
rfc.fit(X_train, y_train_col)

rfc.score(X_test, y_test_col)
#validation croisée
scores = cross_val_score(rfc, X_train, y_train_col, cv=5)

