# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:03:22 2017

@author: valentin
"""

import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from ast import literal_eval

from  fonctions_text import * #fonctions associées
#select_lang, ech_data, text_lower, clean_and_split_text, stopwords_supp_list, snowball_list
#fonctions_text import lemmatize_list, join_word_and_clean, recup_interest_var

#recup path
path = os.path.join(os.path.dirname(os.getcwd()), "mis_en_forme")

#import data
data_text = pd.read_csv(path+"\data_cat_lang_tags_text.csv", sep = ";" )

#create new variable, pour la concatenation du text et de la description
   
data_interest = recup_interest_var(data_text["tags"], data_text["category_id"])
text = [d['X'] for d in data_interest]

#passage en minuscule
data_min_text = text_lower(text)
data_min_text = clean_and_split_text(text)  

#stopwords, lemmatisation et stemming
data_min_text = stopwords_supp_list('english', data_min_text)
data_min_text = lemmatize_list(data_min_text)
data_min_text = snowball_list(data_min_text, 'english')  

#Remise en forme du text nettoyé  
data_fin = join_word_and_clean(data_min_text)

all_data = []
for i in range(len(data_fin)):
    tmp = {"concat_text": data_fin[i], "category_id": data_interest[i]["Y"] }
    all_data.append(tmp)

data_text_fin = pd.DataFrame(all_data)



#concept
data_tags = pd.read_csv(path+"\data_cat_lang_tags_text.csv ", sep = ";", encoding="ISO-8859-1")

#category name 
data_cat = select_category(data_tags, "category_name") 
data_id = select_category(data_tags, "category_id") 
data_corres = []
for i in range(len(data_cat)):
    data = [data_cat[i], data_id[i]]
    data_corres.append(data)

data_corres = pd.DataFrame(data_corres)
#creation de phrase pour récuperer le sens semantique des categories
sent_religion = ['they','have','formed','a','group','to','pray','and','deepend','their','faith']
sent_movie = ['last', 'saunday', 'i', 'went', 'to','see', 'an','action','movie','and','i','love','it']
sent_health = ['my','dad','went','to','the','hospital','to','see','a','doctor','because','he','is','sick']
sent_music = ['i','enjoy','listening','to','pop','and','metal','music']
sent_food =  ['tonight','i','am','going','to','cook','a','good','dinner']
sent_sport = ['Rodger','Federer','is','the','best','tennis','player','of','the','world']
sent_politics = ['Barack','Obama','is','the','new','president','of','the','united','states','of','america']
sent_auto = ['mercedes','benz','introduced','a','brand','new','car']


#Creation d"1 ataframe par catégorie, contenant les concepts wordnet associés à la catégorie
category_1 , synonym_1 , hypernyms_1, hyponyms_1 =  concept(sent_religion, 'religion', data_cat[0])
category1 = {"category_name":category_1[0], "vect_concept": category_1 + synonym_1 + sum(hypernyms_1, []) + sum(hyponyms_1, [])}
cat1 = pd.DataFrame([category1])

category_2 , synonym_2, hypernyms_2, hyponyms_2 =  concept(sent_health, 'medical', data_cat[1])
category2= {"category_name":category_2[0], "vect_concept": category_2 + synonym_2 + sum(hypernyms_2, []) + sum(hyponyms_2, [])}
cat2 = pd.DataFrame([category2])

category_3 , synonym_3, hypernyms_3, hyponyms_3 =  concept(sent_movie, 'movie', data_cat[2])
category3 = {"category_name":category_3[0], "vect_concept": category_3 + synonym_3 + sum(hypernyms_3, []) + sum(hyponyms_3, [])}
cat3 = pd.DataFrame([category3])

category_4 , synonym_4, hypernyms_4, hyponyms_4 =  concept(sent_music, 'music', data_cat[3])
category4 = {"category_name":category_4[0], "vect_concept": category_4 + synonym_4 + sum(hypernyms_4, []) + sum(hyponyms_4, [])}
cat4 = pd.DataFrame([category4])

category_5 , synonym_5, hypernyms_5, hyponyms_5 =  concept(sent_food, 'food', data_cat[4])
category5 = {"category_name":category_5[0], "vect_concept": category_5 + synonym_5 + sum(hypernyms_5, []) + sum(hyponyms_5, [])}
cat5 = pd.DataFrame([category5])

category_6 , synonym_6, hypernyms_6, hyponyms_6 =  concept(sent_sport, 'sport', data_cat[5])
category6= {"category_name":category_6[0], "vect_concept": category_6 + synonym_6 + sum(hypernyms_6, []) + sum(hyponyms_6, [])}
cat6 = pd.DataFrame([category6])

category_7 , synonym_7, hypernyms_7, hyponyms_7 =  concept(sent_politics, 'politics', data_cat[6])
category7 = {"category_name":category_7[0], "vect_concept": category_7 + synonym_7 + sum(hypernyms_7, []) + sum(hyponyms_7, [])}
cat7 = pd.DataFrame([category7])

category_8 , synonym_8, hypernyms_8, hyponyms_8 =  concept(sent_auto, 'vehicle', data_cat[7])
category8 = {"category_name":category_8[0], "vect_concept": category_8 + synonym_8 + sum(hypernyms_8, []) + sum(hyponyms_8, [])}
cat8 = pd.DataFrame([category8])

frames = [cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8]

#resultat final
result_concept = pd.concat(frames)
result_concept = result_concept.reset_index(drop=True)

#nettoyage
for i in range(len(result_concept)):
    for j in range(len(result_concept["vect_concept"][i])):
        result_concept["vect_concept"][i][j] = re.sub('[\W\_]', ' ',  result_concept["vect_concept"][i][j])    
        
for i in range(len(result_concept)):
        result_concept["vect_concept"][i]  = " ".join(result_concept["vect_concept"][i])   
        result_concept["vect_concept"][i] = result_concept["vect_concept"][i].split()

#lemmatisation, stowords and snowball stemming
result_concept = stopwords_supp_df('english', result_concept, 'vect_concept')
result_concept = lemmatize_df(result_concept, 'vect_concept')
result_concept = snowball_df( result_concept, 'vect_concept', 'english')

#suppression des doublons
for i in range(len(result_concept)):
    result_concept["vect_concept"][i] = suppression_doublons_list(result_concept["vect_concept"][i])
  
voc = result_concept["vect_concept"][0] + result_concept["vect_concept"][1] + result_concept["vect_concept"][2] \
+ result_concept["vect_concept"][3] + result_concept["vect_concept"][4] + result_concept["vect_concept"][5] \
+ result_concept["vect_concept"][6] + result_concept["vect_concept"][7]

voc = suppression_doublons_list(voc)



data_vect_count = []
for i in range(len(data_text_fin)):
    cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=voc)
    y = cv.fit_transform([data_text_fin["concat_text"][i]]).toarray()
    data_vect_count.append(y)

for i in range(len(data_vect_count)):
    data_vect_count[i] = data_vect_count[i][0].tolist()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()    

join_vect_concet_id_categ = []
for i in range(len(data_vect_count)):
    tmp = {"vect_concept": data_vect_count[i], "category_id": data_text_fin["category_id"][i], "filename": data_tags["filename"][i]}
    join_vect_concet_id_categ.append(tmp)

join_vect_concept_id_categ = pd.DataFrame(join_vect_concet_id_categ)    

audio = pd.read_csv(path+"\data_audio_mean_vect.csv ", converters={"mean_vect": literal_eval}, sep = ";", decimal=',', encoding="ISO-8859-1")
stat_vid = pd.read_csv(path+"\stat_locuteurs.csv ", sep = ";", decimal=',', encoding="ISO-8859-1")

img = pd.read_csv(path+"\moy_couleur.csv ",converters={"avg_color": literal_eval}, sep = ";", decimal=',', encoding="ISO-8859-1")

feature_img = pd.read_csv(path+"\mdimg.csv",converters={"vecteur_image": literal_eval}, sep = ";", decimal=',', encoding="ISO-8859-1")
vect_stat = []
for i in range(len(stat_vid)):
    tmp = {"filename": stat_vid["filename"][i], "vect_stat": [stat_vid["average_nb_speak"][i], stat_vid["nb_locuteurs"][i],\
                              stat_vid["temps_p"][i]  ]}
    vect_stat.append(tmp)

stat_vid = pd.DataFrame(vect_stat)    
data_audio_txt = pd.merge(join_vect_concept_id_categ ,audio, on='filename')
data_audio_txt_stat = pd.merge(stat_vid ,data_audio_txt, on='filename')
data_audio_txt_stat_img = pd.merge(data_audio_txt_stat ,img, on='filename')

concat_feature_audio_txt = []
for i in range(len(data_audio_txt_stat_img)):
    tmp = {"vect":  softmax(data_audio_txt_stat_img["mean_vect"][i]).tolist()+ softmax(data_audio_txt_stat_img["vect_concept"][i]).tolist()+ softmax(data_audio_txt_stat_img["vect_stat"][i]).tolist()\
           + softmax(data_audio_txt_stat_img["avg_color"][i]).tolist(), "category_id": data_audio_txt_stat_img["category_id_x"][i]}
    concat_feature_audio_txt.append(tmp)

concat_feature_audio_txt = pd.DataFrame(concat_feature_audio_txt)    
X_train, X_test, y_train, y_test = ech_data(concat_feature_audio_txt, "vect", "category_id", 0.33)   

X_train = [X_train.iloc[i] for i in range(len(X_train))]
X_test = [X_test.iloc[i] for i in range(len(X_test))]
y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop=True)

rfc = RandomForestClassifier(n_jobs=-1, oob_score = True) 
 
# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [100, 200, 300, 400, 500],
           "max_depth" : [1, 2,3,4],
           "max_features" : ["auto", "sqrt", "log2", None],
           "min_samples_leaf" : [1, 2, 3, 4 ]}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)

rfc = RandomForestClassifier(n_estimators=100, max_depth=4, max_features=None, min_samples_leaf = 2)
rfc.fit(X_train, y_train)

classif = OneVsRestClassifier(SVC())
classif.fit(X_train, y_train)
rfc.score(X_test, y_test)

rfc.score(X_test, y_test)
scores = cross_val_score(rfc, X_train, y_train, cv=5)
