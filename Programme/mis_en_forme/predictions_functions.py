# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split


def ech_data(data_text, interest_variable, pred_variable, test_size):
    '''Obtention des echantillons d'apprentissage et de test'''                  
    X_train, X_test, y_train, y_test = train_test_split(
        data_text[interest_variable] , data_text[pred_variable],
        test_size= test_size,  random_state=0
    )
    return X_train, X_test, y_train, y_test


def obtain_train_test_elements(data_text, interest_variable, pred_variable, test_size):
    '''Traitement des echantillons d'apprentissage et de test'''
    li_train_test = ech_data(data_text, interest_variable, pred_variable, test_size)
    li_train_test = [
        train_test_element.reset_index(drop=True) for train_test_element in li_train_test
    ]
    X_train, X_test, y_train, y_test = [i for i in li_train_test]
    X_train = [X_train.iloc[i] for i in range(len(X_train))]
    X_test = [X_test.iloc[i] for i in range(len(X_test))]
    return X_train, X_test, y_train, y_test


def estimate_model_prediction(model, X_train, X_test, y_train, y_test):
    '''Resultat du test du predicteur de la prediction'''
    model.fit(X_train, y_train)
    # scores = cross_val_score(model, X_train, y_train, cv=4)
    pred = model.predict(X_test)
    res = []
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res)