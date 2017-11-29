#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:54:07 2017

@author: smr0743a
"""


import os
from os.path import join

def convert_vid_to_wav(path):
    for filename in os.listdir(path):
        if (filename.endswith("flv.ogv")): #or .avi, .mpeg, whatever.
            name = "ffmpeg -i {0}"+" "+str(filename)+".wav"
            os.system(name.format(filename))
        else:
            continue


path = "C:\\Users\\Philippe\\Desktop\\SID\\M2\\Projet_extraction\\Donnees\\DEV_M2SID_VIDEO"

paths = os.listdir(".")
for i in range(len(paths)):
    goal_dir = os.path.join(os.getcwd(), paths[i])
    os.chdir(goal_dir)
    convert_vid_to_wav(goal_dir)
    os.chdir(path)
    
    
