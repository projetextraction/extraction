# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:32:31 2017

@author: valentin
"""

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


from  fonctions_text import * #fonctions associées
#select_lang, ech_data, text_lower, clean_and_split_text, stopwords_supp_list, snowball_list
#fonctions_text import lemmatize_list, join_word_and_clean, recup_interest_var

#recup path
path = os.path.join(os.path.dirname(os.getcwd()), "mis_en_forme")

#import data
data_lang = pd.read_csv(path+"\data_cat_lang_tags_text.csv", sep = ";" )

#filtre langue anglaise
data_text = select_lang(data_lang, 'lang', "eng-usa")

#Remise en forme des index
data_text = data_text.reset_index(drop=True)
    
data_interest = recup_interest_var(data_text["tags"], data_text["category_id"])
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

all_data = []
for i in range(len(data_fin)):
    tmp = {"tags": data_fin[i], "category_id": data_interest[i]["Y"] }
    all_data.append(tmp)

data_tags_fin = pd.DataFrame(all_data)

X_train_tags, X_test_tags, y_train_tags, y_test_tags = ech_data(data_tags_fin, "tags", "category_id", 0.33)   

X_train_tags = X_train_tags.reset_index(drop=True)
y_train_tags = y_train_tags.reset_index(drop=True)
X_test_tags = X_test_tags.reset_index(drop=True)
y_test_tags = y_test_tags.reset_index(drop=True)

#modèle 

#tf_idf
vectorizer = TfidfVectorizer(max_df = 0.95, ngram_range =[1,3])

X_train = vectorizer.fit_transform(X_train_tags)
X_test = vectorizer.transform(X_test_tags)

classif = OneVsRestClassifier(SVC())
classif.fit(X_train, y_train_tags)

scores = cross_val_score(classif, X_train, y_train_tags, cv=5)

pred = classif.score(X_test)

print(pred)





