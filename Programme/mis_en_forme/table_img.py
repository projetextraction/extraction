# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:12:03 2017
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.misc import imread
from glob import glob


        
path = "IMAGE\DEV_M2SID_SHOT"

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
        
table_img = recup_path_img(path)

#exemple of img importation an visualisation
img = imread(table_img["path_img"][71])
plt.imshow(img, cmap='Greys_r')



