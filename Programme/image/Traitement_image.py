# -*- coding: utf-8 -*-

#Importation des packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import matplotlib.image as mpimg
import os
import pandas as pd
import re 
from PIL import Image
from copy import copy
from PIL import Image
from pylab import *


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

plt.show()

#Histogramme de couleur simplifié
img = Image.open('C:/Users/Admin/Desktop/Cours/Master Big Data/Projet M2 texte-audio-image/DEV_M2SID_SHOT/Aabbey1-ShalomHartmanInstitutesInnovativeTheologyConference661/3061.jpg')
img
im = img.convert('L')
im
a=np.array(im)

hist(a.flatten(), 128)
show()

#Récupére rcontour de l'image
contour(im, origin='image')
axis('equal')
axis('off')

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

average_color = calcul_couleur_frequente(img)
print(average_color)


#Récuperation csv couleur dominante vidéo 
path = "C:\\Users\\Admin\\Desktop\\Cours\\Master Big Data\\Projet M2 texte-audio-image\\DEV_M2SID_SHOT"

average_colors = []
for repo in os.listdir(path):
    avgs_colors = []
    for img_file in os.listdir(path + '/' + repo):
        goal_dir = path + '/' + repo + '/' + img_file
        im = Image.open(goal_dir)
        im = im.resize((100,100))
        avgs_colors.append(calcul_couleur_frequente(im))
    mat_avg_colors  = np.asmatrix(avgs_colors)
    mean_avg_colors = mat_avg_colors.sum(axis=0) / mat_avg_colors.shape[0]
    average_colors.append({'filename': repo, 'avg_color': mean_avg_colors.tolist()[0]})
df_average_colors = pd.DataFrame(average_colors)
#         w, h = im.size  
#         colors[i][j] = im.getcolors(w*h)
df_average_colors.to_csv('C:/Users/Admin/Desktop/moy_couleur.csv', sep=';', decimal=',')  


