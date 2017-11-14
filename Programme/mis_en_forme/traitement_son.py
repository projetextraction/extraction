# -*- coding: utf-8 -*-

import os
from bs4 import BeautifulSoup as Soup

import pandas as pd


from table_img import recup_path_img
from tablemedata import extract_metadata, extract_file_category


TRANS_FILE = "../../Data/AUDIO/DEV_M2SID_LIMSI_ASR"
PATH = "../../Data/Video/DEV_M2SID_SHOT"

EXCLUDED_WORDS = [' {fw} ']


def extract_mots(mots, speech_seg, pk_mot, pk_segment):
    words_list = speech_seg.find_all('Word')
    for word in words_list:
        mot = {
            'pk_mot': pk_mot,
            'pk_segment': pk_segment,
            'txt': word.string,
            'duree': word.get('dur'),
            'conf': word.get('conf')
        }
        pk_mot += 1
        mots.append(mot)
    return mots, pk_mot


def extract_segments(segments, mots, sp, pks_locuteur, sp_kids, pk_segment, pk_mot):
    speech_segs = sp.find_all('SpeechSegment')
    for speech_seg in speech_segs:
        segment = {
            'pk_segment': pk_segment,
            'stime': speech_seg.get('stime'),
            'etime': speech_seg.get('etime'),
            'lang': speech_seg.get('lang')
        }
        segment['speaker'] = speech_seg.get('spkid')
        ####
        idx = sp_kids.index(segment['speaker'])
        segment['pk_locuteur'] = pks_locuteur[idx]
        ####
        mots, pk_mot = extract_mots(mots, speech_seg, pk_mot, segment['pk_segment'])
        if segment != {}:
            segments.append(segment)
            pk_segment += 1
    return segments, mots, pk_segment, pk_mot


# vecteurs locuteurs: gender, temps_parole, nombre de speechsegments,
#                     temps moyen de paroles


def extract_locuteurs(document, locuteurs, segments, mots, sp, pk_document, pk_locuteur, pk_segment, pk_mot):
    speakers = sp.find_all('Speaker')
    pks_locuteur , spkids = [], []
    for speaker in speakers:
        locuteur = {
            'pk_locuteur': pk_locuteur,
            'gender': speaker.get('gender'),
            'speaker': speaker.get('spkid'),
            'temps_parole': speaker.get('dur'),
            'pk_document': pk_document
        }
        pks_locuteur.append(pk_locuteur)
        spkids.append(locuteur['speaker'])
        pk_locuteur += 1
        locuteurs.append(locuteur)
        document['lang'] = speaker.get('lang')
    segments, mots, pk_segment, pk_mots =  extract_segments(
        segments, mots, sp, pks_locuteur, spkids, pk_segment, pk_mot
    )
    return document, locuteurs, segments, mots, pk_locuteur, pk_segment, pk_mot


def extract_documents():
    documents = []
    locuteurs, segments, mots = [], [], []
    pk_document = 1
    pk_locuteur, pk_segment, pk_mot = 1, 1, 1
    for trans_file in os.listdir(TRANS_FILE):
        document = {}
        xml_file = open(TRANS_FILE + '/' + trans_file, encoding = "UTF-8")
        xml_r = xml_file.read()
        sp = Soup(xml_r, 'xml')
        if trans_file[-3:] == 'xml':
            file_name = trans_file[0:len(trans_file)-12]
        document = {
            'pk_document': pk_document,
            'filename': file_name
        }
        document, locuteurs, segments, mots, pk_locuteur, pk_segment, pk_mots =  extract_locuteurs(
            document, locuteurs, segments, mots, sp, pk_document, pk_locuteur, pk_segment, pk_mot
        )
        pk_document += 1
        documents.append(document)
    return documents, locuteurs, segments, mots


def extract_text_doc(doc_title):
    text_doc = []
    dfa = df.loc[df['filename'] == doc_title]
    dfb = dfa.sort_values(['pk_mot'], ascending=True)
    for pk_seg in dfb['pk_segment'].unique():
        dfc = dfb.loc[df['pk_segment'] == pk_seg]
        for mot in list(dfc['txt']):
            if mot not in EXCLUDED_WORDS:
                text_doc.append(mot)
    return text_doc

doc_title = 'Aabbey1-InvitationToSummer2009RabbinicalStudySeminarAtHartmanIn262'
# ' '.join(liste)


documents, locuteurs, segments, mots = extract_documents()
table_img = recup_path_img(PATH)
df_metadata, df_data_tags, df_data_user = extract_metadata()
df_category = extract_file_category()
df_documents, df_locuteurs, df_segments, df_mots = (
    pd.DataFrame(documents), pd.DataFrame(locuteurs),
    pd.DataFrame(segments), pd.DataFrame(mots)
)

df = df_documents
dataframes = [
    #[table_img, 'filename'], [df_category, 'filename'],
    [df_locuteurs, 'pk_document'], [df_segments, 'pk_locuteur'], [df_mots, 'pk_segment'],
    # [df_metadata, 'filename'], [df_data_tags, 'filename'], [df_data_user, 'filename']
]
for dtf, join_keys in dataframes:
    df = pd.merge(df, dtf, how='inner', on=join_keys)

df.to_csv('data_met_cat.csv', sep=';', decimal=',')


'''
df0 = pd.merge(df_documents, table_img, how='inner', on='filename')
df1 = pd.merge(df_documents, df_locuteurs, how='inner', on='pk_document')
df2 = pd.merge(df1, df_segments, how='inner', on='pk_locuteur')
df3 = pd.merge(df2, df_mots, how='inner', on='pk_segment')
'''