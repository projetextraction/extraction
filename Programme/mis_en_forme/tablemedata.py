# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:29:19 2017

@author: Valentin
"""
import pandas as pd
import os
from bs4 import BeautifulSoup as Soup
# from sonarqube_api import SonarAPIHandler


TXT_REPOSITORY = '../../Data/Txt/DEV_M2SID_METADATA/'
CATEGORY_REPOSITORY = '../../../DEV_M2SID_VIDEO/'
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


def extract_file_category():
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


def table_metadata(sp):
    description = sp.find_all('description')[0].string
    duree = sp.find_all('duration')[0].string
    url = sp.find_all('url')[0].string
    title = sp.find_all('title')[0].string
    filename = sp.find_all('filename')[0].string
    # remove '.flv'
    filename = filename[0:len(filename)-4]
    link = sp.find_all('link')[0].string
    size = sp.find_all('size')[0].string
    return description, duree, url, title, filename, link, size

def table_tags(sp):
    tags = []
    tmp_tags = sp.find_all('string')
    for i in range(len(tmp_tags)):
        tag = sp.find_all('string')[i].string
        tags.append(tag)
    return tags

def table_user(sp):
    uid = sp.find_all('uid')[0].string
    login = sp.find_all('login')[0].string
    return uid, login


def extract_metadata():
    filenames = os.listdir(TXT_REPOSITORY)
    metadata, data_tags, data_user = [], [], []
    for filename in filenames: 
        data = open(TXT_REPOSITORY + filename).read()
        sp = Soup(data, 'xml')
        metadat = table_metadata(sp)
        xml_meta_data = {
            "description": metadat[0],
            "duree": metadat[1],
            "url": metadat[2],
            "title": metadat[3],
            "filename": metadat[4],
            "link": metadat[5], 
            "size": metadat[6]
        }
        tags = {"tags": table_tags(sp), "filename": metadat[4]}
        tab_usr = table_user(sp)
        user_tmp = {
            "uid": tab_usr[0],
            "login": tab_usr[1],
            "filename": metadat[4]
        }
        metadata.append(xml_meta_data)
        data_tags.append(tags)
        data_user.append(user_tmp)
    df_metadata, df_data_tags, df_data_user = (
        pd.DataFrame(metadata), pd.DataFrame(data_tags), pd.DataFrame(data_user)
    )
    return df_metadata, df_data_tags, df_data_user

#return records as DataFrame


# metadata, data_tags, data_user = extract_metadata()




# tags = sp.find_all('string')