# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:12:03 2017
"""

img = open("C:/Users/Philippe/Desktop/SID/M2/Projet_extraction/Donnees/DEV_M2SID_SHOT/Aabbey1-InvitationToSummer2009RabbinicalStudySeminarAtHartmanIn262/012.jpg",'wb')

import urllib.request

urllib.request.urlretrieve("https://github.com/projetextraction/extraction/012.jpg")
