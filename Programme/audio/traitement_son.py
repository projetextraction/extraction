# -*- coding: utf-8 -*-

import pandas as pd
from ast import literal_eval

from table_img import recup_path_img
from tablemedata import extract_metadata, extract_file_category
from table_audio import (
    extract_documents, extract_texts, calc_all_files_distinct_speaking_periods,
    obtain_mfccs, delete_mfcc_empty_element, obtain_df_mean_vect
)
from predictions_functions import (
    obtain_train_test_elements, estimate_model_prediction
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

PATH = "../../Data/Video/DEV_M2SID_SHOT"


def merge_dataframes(base_dataframe, other_dataframes):
    '''Merge a dataframe with other(s)'''
    # The other dataframe is 2-dimensionnal element with dataframe and join key
    df = base_dataframe
    for dtf, join_keys in other_dataframes:
        df = pd.merge(df, dtf, how='inner', on=join_keys)
    return df


# Calcul de tous les dataframes "de base"
df_documents, df_locuteurs, df_segments, df_mots = extract_documents()
table_img = recup_path_img(PATH)
df_metadata, df_data_tags, df_data_user = extract_metadata()
df_category = extract_file_category()

# Calcul du dataframe contenant les textes reconstitués
dfx = merge_dataframes(df_documents, [
    [df_locuteurs, 'pk_document'], [df_segments, 'pk_locuteur'], [df_mots, 'pk_segment']
])
df_texts = extract_texts(dfx)

# Extraction de dataframe(s) et stockage au format csv pour faire des analyses
'''
dfy = merge_dataframes(df_documents, [
    [df_category, 'filename'], [df_metadata[['filename', 'description']], 'filename'],
    [df_data_tags, 'filename'], [df_texts, 'filename']
])
'''
'''dfy.to_csv('data_cat_lang_tags_text.csv', sep=';', decimal=',')'''

# Calcul du dataframe contenant les intervalles de paroles et de non paroles
# pour un fichier audio donné
dfz = merge_dataframes(df_documents, [
    [df_locuteurs, 'pk_document'], [df_segments, 'pk_locuteur']
])
df_speak = calc_all_files_distinct_speaking_periods(dfz, df_documents)

# Calcul du mfcc moyen pour la non parole et association avec la categorie
df_mfcc_filtre = obtain_mfccs(df_speak)
df_mfcc_filtre = delete_mfcc_empty_element(df_mfcc_filtre)
df_mean_vect = obtain_df_mean_vect(df_mfcc_filtre)
df_audio = merge_dataframes(df_mean_vect, [
    [df_category, 'filename']
])
'''df_audio.to_csv('data_audio_mean_vect.csv', sep=';', decimal=',')'''
'''df_audio = pd.read_csv(
    "data_audio_mean_vect.csv", converters={"mean_vect": literal_eval},
    sep=';', decimal=','
)'''

# Obtention des echantillons d'apprentissage et de test
X_train, X_test, y_train, y_test = obtain_train_test_elements(
    data_text=df_audio, interest_variable='mean_vect',
    pred_variable='category_name', test_size=0.33
)

# Definition du modele
'''classif = OneVsRestClassifier(SVC())'''
rfc = RandomForestClassifier(n_jobs=-1, oob_score = True)
# Use a grid over parameters of interest
param_grid = {
    "n_estimators" : [100, 200],
    "max_depth" : [1, 2, 3, 4],
    "max_features" : ["auto", "sqrt", "log2", None],
    "min_samples_leaf" : [1, 2, 3, 4]
}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=4)
estimate_model_prediction(CV_rfc, X_train, X_test, y_train, y_test)


model = RandomForestClassifier(
    n_estimators=200, criterion='gini', max_depth=4, min_samples_split=2,
    min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features=None,
    max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
    bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
    warm_start=False, class_weight=None
)

# Estimation de la prediction du modele
print(estimate_model_prediction(model, X_train, X_test, y_train, y_test))
'''print(estimate_model_prediction(classif, X_train, X_test, y_train, y_test))'''