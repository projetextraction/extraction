# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:22:32 2017

@author: valentin
"""
import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from  fonctions_text import * #fonctions associées
#select_lang, ech_data, text_lower, clean_and_split_text, stopwords_supp_list, snowball_list
#fonctions_text import lemmatize_list, join_word_and_clean, recup_interest_var

#recup path
path = os.path.join(os.path.dirname(os.getcwd()), "mis_en_forme")

#import data
data_lang = pd.read_csv(path+"\data_cat_lang_tags_text.csv", sep = ";" )

#create new variable, pour la concatenation du text et de la description
data_lang["concat_text"] = ""

#filtre langue anglaise
data_text = select_lang(data_lang, 'lang', "eng-usa")

#Remise en forme des index
data_text = data_text.reset_index(drop=True)

#remplissage variabke créée
for i in range(len(data_text)):
    data_text["concat_text"][i] = str(data_text["description"][i]) +' ' +str(data_text["text"][i])

    
data_interest = recup_interest_var(data_text["concat_text"], data_text["category_id"])
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

X_train_concat_text, X_test_concat_text, y_train_concat_text, y_test_concat_text = ech_data(data_text_fin, "concat_text", "category_id", 0.33)   

X_train_concat_text = X_train_concat_text.reset_index(drop=True)
y_train_concat_text = y_train_concat_text.reset_index(drop=True)

X_test_concat_text = X_test_concat_text.reset_index(drop=True)
y_test_concat_text = y_test_concat_text.reset_index(drop=True)

#modèle 

#tf_idf
vectorizer = TfidfVectorizer(max_df = 0.95, ngram_range =[1,3])

X_train = vectorizer.fit_transform(X_train_concat_text)
X_test = vectorizer.transform(X_test_concat_text)

model_to_set = OneVsRestClassifier(SVC())
model_to_set.fit(X_train, y_train_concat_text)

scores = cross_val_score(model_to_set, X_train, y_train_concat_text, cv=5)

pred = model_to_set.score(X_test)

print(pred)






