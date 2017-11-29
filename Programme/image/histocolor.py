# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:16:40 2017

@author: Cheikh Rouhou.H
"""

import struct
from PIL import Image
import scipy
import scipy.misc
import scipy.cluster
import os
from os.path import join
#
#def convert_vid_to_wav(path):
#    for filename in os.listdir(path):
#        if (filename.endswith("flv.ogv")): #or .avi, .mpeg, whatever.
#            name = "ffmpeg -i {0}"+" "+str(filename)+".wav"
#            os.system(name.format(filename))
#        else:
#            continue
#
#path = "'Z:/DEV_M2SID_SHOT/Aabbey1-InvitationToSum"
#
#paths = os.listdir(".")
#for i in range(len(paths)):
#    goal_dir = os.path.join(os.getcwd(), paths[i])
#    os.chdir(goal_dir)
#    convert_vid_to_wav(goal_dir)
#    os.chdir(path)
#    
    
# CHisto coleur   


import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import matplotlib.image as mpimg

img = imread('z:/chh9641a/Desktop/3061.jpg')
#plt.imshow(img, cmap='Greys_r')


width, height = img.size
w, h = img.size
pixels = img.getcolors(w * h)
pixel_frequent = pixels[0]

#for count, colour in pixels:
#        if count > pixel_frequent[0]:
#            pixel_frequent = (count, colour)
#            
#pixel_frequent


from PIL import Image

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

    return (r_total/count, g_total/count, b_total/count)

img = Image.open('z:/chh9641a/Desktop/3061.jpg')
#img = img.resize((150,150))  
average_color = calcul_couleur_frequente(img)
print(average_color)


w, h = img.size  
colors = img.getcolors(w*h)

img = imread('z:/chh9641a/Desktop/3061.jpg')
s = img.shape


def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)
    
for idx, c in enumerate(colors):
        plt.bar(idx, c[0], color=hexencode(c[1]),edgecolor=hexencode(c[1]))

plt.show()


img = imread('z:/chh9641a/Desktop/3061.jpg')
s = img.shape

def seuil(image,limite):
    resultat = image.copy()
    s = image.shape
    for j in range(s[0]):
        for i in range(s[1]):
            if image[j,i] > limite:
               resultat[j,i] = 255
            else:
                resultat[j,i] = 0
    return resultat
 
img = imread('z:/chh9641a/Desktop/3061.jpg')
img2 = seuil(img,150)




import sys, os
import cv2

def detecte_visages(image, image_sortie, show = False):
    img = cv2.imread(image)
    face_model = cv2.CascadeClassifier("z:/chh9641a/Desktop/3061.jpg")
    faces = face_model.detectMultiScale(img)
    print ("nombre de visages", len(faces), "dimension de l'image", img.shape, "image", image)
    for face in faces:
        cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[0] + face[3]), (255, 0, 0), 3)
        
    cv2.imwrite(image_out, img)

    if show :
        cv2.imshow("visage",img)
        if cv2.waitKey(5000) == 27: cv2.destroyWindow("visage")
   
#if __name__ == "__main__":
#    # applique 
#    for file in os.listdir(".") :
#        if file.startswith("visage") : continue # déjà traité
#        if os.path.splitext(file)[-1].lower() in [".jpg", ".jpeg", ".png" ] :
#            detecte_visages (file, "visage_" + file)




