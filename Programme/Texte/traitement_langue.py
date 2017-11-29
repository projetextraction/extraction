# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:23:02 2017

@author: smr0743a
"""

import re 
import os
import numpy as np
import pandas as pd


from  fonctions_text import * #fonctions associées

import sklearn

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer



path = os.path.join(os.path.dirname(os.getcwd()), "mis_en_forme")
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

#import tags
data_tags= pd.read_csv(path+"\data_cat_lang_tags_text.csv", sep = ";", encoding="ISO-8859-1")

data_tags_concept = pd.merge(data_tags , result_concept, on='category_name')
data_tags_concept = data_tags_concept.sort_values(['pk_document'], ascending = True)
#Remise en forme des index
data_tags_concept = select_lang(data_tags_concept, 'lang', "eng-usa")
data_tags_concept = data_tags_concept.reset_index(drop=True)

    
data_interest = recup_interest_var(data_tags_concept["tags"], data_tags_concept["category_name"])

#recuperation d'une liste de tous les tags
tags = [d['X'] for d in data_interest]

#passage en minuscule
data_min_tags = text_lower(tags)
data_min_tags = clean_and_split_text(tags)  

#stopwords, lemmatisation et stemming
data_tags = stopwords_supp_list('english', data_min_tags)
data_tags = lemmatize_list(data_tags)
data_tags = snowball_list(data_tags, 'english')  

#Remise en forme du text nettoyé  
data_fin = join_word_and_clean(data_tags)

#split
for i in range(len(data_fin)):
    data_fin[i] = data_fin[i].split()
    
#recuperation tags et category naùe avant jointure
all_data = []
for i in range(len(data_fin)):
    tmp = {"tags": data_fin[i], "category_name": data_interest[i]["Y"] }
    all_data.append(tmp)
    
#convert to dataframe
data_tags_fin = pd.DataFrame(all_data)

#merge
data_tags_concept_fin = pd.merge(data_tags_fin , data_tags_concept, on='category_name')
#suppression doublons
data_tags_concept_fin = data_tags_concept_fin.drop_duplicates(['filename'], keep='last') 
#tri par clé des documents
data_tags_concept_fin = data_tags_concept_fin.sort_values(['pk_document'], ascending = True)
#reset index
data_tags_concept_fin = data_tags_concept_fin.reset_index(drop=True)

#apprentissage et test
X_train_tags, X_test_tags, y_train_tags, y_test_tags = ech_data(data_tags_concept_fin, "tags_x", "category_name", 0.33)   

#reset index
X_test_tags = X_test_tags.reset_index(drop=True)
y_test_tags = y_test_tags.reset_index(drop=True)

X_train_tags = X_train_tags.reset_index(drop=True)
y_train_tags = y_train_tags.reset_index(drop=True)

#vecteur de tags trains par categorie : 1 vecteur = liste contenant tous les tags du train associée à la categorie
concat_train = pd.concat([
X_train_tags, y_train_tags], axis=1)
    
concat_train_list = concat_train.groupby('category_name')['tags_x'].apply(list)
x = pd.DataFrame(concat_train_list)

#récuperation des valeurs du tableau
tab = []
tmp = {}
for i in range(len(x)):
    for j in range(len(x.iloc[i])):
        tmp = {"tags": x.iloc[i][j], "id":i}
        tab.append(tmp)

#création d'une liste de tags     
concat_tags = []        
for i in range(len(tab)):
    y = {"tags":' '.join([str(x) for x in tab[i]["tags"]]), "id":tab[i]["id"]}
    concat_tags.append(y)

#re-nettoyage des tags
for i in range(len(concat_tags)):  
        concat_tags[i]["tags"] = re.sub('[\W\_]', ' ', concat_tags[i]["tags"])
        concat_tags[i]["tags"] = re.sub('nan', '', concat_tags[i]["tags"]) 
        concat_tags[i]["tags"] = concat_tags[i]["tags"].split()

#table de correspondance  categorie name
data_corres = data_corres.sort_index(by=[1], ascending=[True])
data_corres = data_corres.reset_index(drop=True)

#recuperation du vecteur de tags nettoyé par categorie
id_cat_tags = []        
for i in range(len(concat_tags)):
    recup_id_cat = {"category_name":data_corres[0][i], "tags":concat_tags[i]['tags']}
    id_cat_tags.append(recup_id_cat)

id_cat_tags = pd.DataFrame(id_cat_tags)

#merge to have concept
data_train_tags_fin = pd.merge(result_concept,id_cat_tags, on='category_name')

#concatenation des concepts wordnet et des listes de tags par categorie
final_train_vect = []
for i in range(len(data_train_tags_fin)):
    tmp_final_vect = {"concept_tags":data_train_tags_fin["tags"][i] + data_train_tags_fin["vect_concept"][i], "category_name":data_train_tags_fin["category_name"][i]}
    final_train_vect.append(tmp_final_vect)


#Jaccard similarity for all categories, between train (tags and concepts) and test (just tags)
score_test = []
for i in range(len(X_test_tags)):
    for j in range(len(final_train_vect)):
           tmp = {"score":float(len(intersect(final_train_vect[j]["concept_tags"], X_test_tags[i]))) / len(union(final_train_vect[j]["concept_tags"], X_test_tags[i])), "category_name":final_train_vect[j]["category_name"], "id_test":i}
           score_test.append(tmp)
           
score_test  = pd.DataFrame(score_test)
 
#max score by test        
idx = score_test.groupby('id_test')['score'].idxmax()



def using_idxmax_loc(df):
    idx = df.groupby('id_test')['score'].idxmax()
    return df.loc[idx, ['category_name', 'score']]

#reste index
result = using_idxmax_loc(score_test).reset_index(drop=True)

#if max score > 0 -> similarité detectée = on rattache le, jeu de test à la categorie correspondant au score max
result_for_tags = result[result["score"] > 0
result_for_tags = result_for_tags.reset_index(drop = True)

#sinon : aucune categorie attribuée, on recupere les id correspondants, pour alimenter le modele "description texte"
#pour ameliorer la prediction
result_for_model = result[result["score"] == 0]

#result pred part 1 (on doit utiliser l'autre modele pour avoir le score totale)
res = []
for i in range(len(result_for_tags )):
    if result_for_tags ["category_name"][i] == y_test_tags[result_for_tags.iloc[i].name]:
        y = 1
        res.append(y)
    else:
        y=0
sum(res)/len(res)

#-------------------------------------Modele description-text ----------------------------------------------------------------------------

#recup path
path = os.path.join(os.path.dirname(os.getcwd()), "mis_en_forme")

#import data
data_lang = pd.read_csv(path+"\data_cat_lang_tags_text.csv", sep = ";" )
    
#modele description et text       
data_lang["concat_text"] = ""

#filtre langue anglaise
data_text = select_lang(data_lang, 'lang', "eng-usa")

#Remise en forme des index
data_text = data_text.reset_index(drop=True)

#remplissage variabke créée
for i in range(len(data_text)):
    data_text["concat_text"][i] = str(data_text["description"][i]) +' ' +str(data_text["text"][i])

    
data_interest = recup_interest_var(data_text["concat_text"], data_text["category_name"])
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
    tmp = {"concat_text": data_fin[i], "category_name": data_interest[i]["Y"] }
    all_data.append(tmp)

data_text_fin = pd.DataFrame(all_data)

X_train_concat_text, X_test_concat_text, y_train_concat_text, y_test_concat_text = ech_data(data_text_fin, "concat_text", "category_name", 0.33)   

X_train_concat_text = X_train_concat_text.reset_index(drop=True)
y_train_concat_text = y_train_concat_text.reset_index(drop=True)
X_test_concat_text = X_test_concat_text.reset_index(drop=True)
y_test_concat_text = y_test_concat_text.reset_index(drop=True)

result_for_model = result[result["score"] == 0]

data_X_to_pred = []
for i in range(len(result_for_model)):
    ll = result_for_model.iloc[i].name
    data_X_to_pred.append(ll)
    
X_test_data_pred = []
for i in data_X_to_pred:
    data_to_pred = X_test_concat_text[i]
    X_test_data_pred.append(data_to_pred)

y_test_data_pred = []    
for i in data_X_to_pred:
    data_to_pred = y_test_concat_text[i]
    y_test_data_pred.append(data_to_pred)


#modèle 

#tf_idf
vectorizer = TfidfVectorizer(max_df = 0.95, ngram_range =[1,3])

X_train = vectorizer.fit_transform(X_train_concat_text)
X_test = vectorizer.transform(X_test_data_pred)

#classifier
classif = OneVsRestClassifier(SVC())
classif.fit(X_train, y_train_concat_text)

pred = classif.predict(X_test)
    
#result into list
res2 = []
for i in range(len(pred)):
    if pred[i] == y_test_concat_text[data_X_to_pred[i]]:
        y = 1
        res2.append(y)
    else:
        y=0
        res2.append(y)

#resultat final, concatenation du resultat semantique et du resultat du classifieur
res_fin = res+res2

print("final_score "+ str(sum(res_fin)/len(res_fin)))
