# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:37:55 2017

@author: valentin
"""
import re

from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.cross_validation import train_test_split




def select_lang(data, variable, lang):
    data_text = data[data[variable].str.contains(lang)==True]
    return data_text


def ech_data(data_text, interest_variable, pred_variable, test_size):                    
    X_train, X_test, y_train, y_test = train_test_split(
        data_text[interest_variable] , data_text[pred_variable], test_size= test_size,  random_state=0)
    return X_train, X_test, y_train, y_test


def text_lower(X_train):
    data_min = []
    for i in range(len(X_train)):
        X_train[i] = X_train[i]
        minuscule = X_train[i].lower()
        data_min.append(minuscule)
    return data_min


def clean_and_split_text(data_min):
    for i in range(len(data_min)):    
        data_min[i]= re.sub('[\W\_]', ' ', data_min[i])
        data_min[i]= re.sub('\sp\s', ' ', data_min[i])
        data_min[i]= re.sub('nan', ' ', data_min[i])
        data_min[i]= re.sub('www', ' ', data_min[i])
        data_min[i]= re.sub('http', ' ', data_min[i])
        data_min[i]= re.sub('https', ' ', data_min[i])
        data_min[i]= re.sub('href', ' ', data_min[i])
        data_min[i]= re.sub('\sorg\s', ' ', data_min[i])
        data_min[i] = data_min[i].split()
    return data_min


def stopwords_supp_list(language, list_values):
    stopWords = set(stopwords.words(language))
    for i in range(len(list_values)):
        list_values[i] = [w for w in list_values[i] if w not in stopWords] 
    return list_values


def lemmatize_list(list_values):
    wordnet_lemmatizer = WordNetLemmatizer()
    for i in range(len(list_values)):
        for j in range(len(list_values[i])):
            list_values[i][j] =  wordnet_lemmatizer.lemmatize(list_values[i][j])
    return list_values


def snowball_list(list_values, language):
    snowball_stemmer = SnowballStemmer(language)   
    for i in range(len(list_values)):
        for j in range(len(list_values[i])):
            list_values[i][j] =   snowball_stemmer.stem(list_values[i][j])
    return list_values  

def join_word_and_clean(data_min):        
    data_app = []        
    for i in range(len(data_min)):    
        tr = " ".join(data_min[i])
        data_app.append(tr)
    return data_app


def recup_interest_var(df_X, df_Y):
    data_interest = []
    for i in range(len(df_X)):
        tmp = {"Y": df_Y[i], "X": df_X[i]}
        data_interest.append(tmp)
    return data_interest


def select_category(df, interest_var):
    df_categ = df[interest_var].unique()
    return df_categ


def concept(sentence, interest_word, category_name):
    cat = [category_name]
    sent = lesk(sentence, interest_word, 'n')
    synonym = sent.lemma_names()
    hyp = sent.hypernyms()
    hyponyms = sent.hyponyms()
    hyp_lst = []
    hypo_lst = []
    for i in range(len(hyp)):
        hyp_tmp = hyp[i].lemma_names()
        hyp_lst.append(hyp_tmp)
    for j in range(len(hyponyms)):
        hypo_tmp = hyponyms[j].lemma_names()
        hypo_lst.append(hypo_tmp)
    return cat, synonym, hyp_lst, hypo_lst


def suppression_doublons_list(list_values):
    list_without_doub = list(set(list_values))
    return list_without_doub


def stopwords_supp_df(language, df, variable):
    stopWords = set(stopwords.words(language))
    for i in range(len(df[variable])):
        df[variable][i] = [w for w in df[variable][i] if w not in stopWords] 
    return df


def lemmatize_df(df, variable):
    wordnet_lemmatizer = WordNetLemmatizer()
    for i in range(len(df[variable])):
        for j in range(len(df[variable][i])):
            df[variable][i][j] =  wordnet_lemmatizer.lemmatize(df[variable][i][j])
    return df

def snowball_df(df, variable, language):
    snowball_stemmer = SnowballStemmer(language)   
    for i in range(len(df[variable])):
        for j in range(len(df[variable][i])):
            df[variable][i][j] = snowball_stemmer.stem(df[variable][i][j])
    return df      

def intersect(a, b):
    return list(set(a) & set(b))
    
def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


