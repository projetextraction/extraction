# -*- coding: utf-8 -*-

import os
from bs4 import BeautifulSoup as Soup
from unidecode import unidecode
import pandas as pd
import numpy as np
import wave

from tablemedata import TXT_REPOSITORY
from tp_audio import mfcc, vad_filter


TRANS_FILE = "../../Data/AUDIO/DEV_M2SID_LIMSI_ASR"
AUDIO_REPOSITORY = "../../Data/Audio/DEV_M2SID_LIMSI_ASR"
WAVE_FILES = "../../../sons.wav/"

EXCLUDED_WORDS = [' {fw} ']


def extract_mots(mots, speech_seg, pk_mot, pk_segment):
    '''Extraction des mots reconnus d'un segment de parole'''
    words_list = speech_seg.find_all('Word')
    for word in words_list:
        mot = {
            'pk_mot': pk_mot,
            'pk_segment': pk_segment,
            'txt': unidecode(word.string),
            'duree': word.get('dur'),
            'conf': word.get('conf')
        }
        pk_mot += 1
        mots.append(mot)
    return mots, pk_mot


def extract_segments(segments, mots, sp, pks_locuteur, sp_kids, pk_segment, pk_mot):
    '''Extration des segments de paroles enonce par un locuteur contenant des mots'''
    speech_segs = sp.find_all('SpeechSegment')
    for speech_seg in speech_segs:
        segment = {
            'pk_segment': pk_segment,
            'stime': speech_seg.get('stime'),
            'etime': speech_seg.get('etime'),
            'lang': speech_seg.get('lang')
        }
        # stime et etime correspondent aux debuts et aux fins de segments de parole
        spker = speech_seg.get('spkid')
        idx = sp_kids.index(spker)
        segment['pk_locuteur'] = pks_locuteur[idx]
        mots, pk_mot = extract_mots(mots, speech_seg, pk_mot, segment['pk_segment'])
        if segment != {}:
            segments.append(segment)
            pk_segment += 1
    return segments, mots, pk_segment, pk_mot


def extract_locuteurs(document, locuteurs, segments, mots, sp, pk_document,
pk_locuteur, pk_segment, pk_mot):
    '''Extraction des informations sur les locuteurs d'un document audio'''
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
    '''Extraction des documents'''
    documents = []
    locuteurs, segments, mots = [], [], []
    pk_document = 1
    pk_locuteur, pk_segment, pk_mot = 1, 1, 1
    for trans_file in os.listdir(TRANS_FILE):
        xml_file = open(TRANS_FILE + '/' + trans_file, encoding = "UTF-8")
        xml_r = xml_file.read()
        sp = Soup(xml_r, 'xml')
        if trans_file[-3:] == 'xml':
            file_name = trans_file[0:len(trans_file)-12]
        channel = sp.find_all('Channel')[0]
        # il y a toujours un channel
        document = {
            'pk_document': pk_document,
            'filename': file_name,
            'duration': channel.get('sigdur')
            # duree du signal
        }
        document, locuteurs, segments, mots, pk_locuteur, pk_segment, pk_mots =  extract_locuteurs(
            document, locuteurs, segments, mots, sp, pk_document, pk_locuteur, pk_segment, pk_mot
        )
        pk_document += 1
        documents.append(document)
    df_documents, df_locuteurs, df_segments, df_mots = (
        pd.DataFrame(documents), pd.DataFrame(locuteurs),
        pd.DataFrame(segments), pd.DataFrame(mots)
    )
    return df_documents, df_locuteurs, df_segments, df_mots


def extract_text_doc(doc_title, df):
    text_doc = []
    dfa = df.loc[df['filename'] == doc_title]
    dfb = dfa.sort_values(['pk_mot'], ascending=True)
    for pk_seg in dfb['pk_segment'].unique():
        dfc = dfb.loc[df['pk_segment'] == pk_seg]
        for mot in list(dfc['txt']):
            if mot not in EXCLUDED_WORDS:
                text_doc.append(unidecode(mot))
    return text_doc


def extract_texts(df):
    texts_list = []
    for file in os.listdir(TXT_REPOSITORY):
        filename = file[0:len(file)-4]
        text = {
            'filename': filename,
            'text': ''.join(extract_text_doc(filename, df))
        }
        texts_list.append(text)
    texts = pd.DataFrame(texts_list)
    return texts


def distinct_speaking_periods(filename, dfx, df_documents):
    speaking_periods, not_speaking_periods = [], []
    dfa = dfx.loc[dfx['filename'] == filename]
    begin_not_speaking_period = 0.0
    if dfa.shape[0] == 0:
        dfd = df_documents.loc[df_documents['filename'] == filename]
        not_speaking_periods.append(
            [begin_not_speaking_period, dfd.iloc[0]['duration']]
        )
    else:
        dfb = dfa.sort_values(['pk_segment'], ascending=True)
        for i in range(dfb.shape[0]):
            if i != dfb.shape[0]:
                not_speaking_periods.append(
                    [begin_not_speaking_period, float(dfb.iloc[i]['stime'])]
                )
                begin_not_speaking_period = float(dfb.iloc[i]['etime'])
                speaking_periods.append([float(dfb.iloc[i]['stime']), begin_not_speaking_period])
        not_speaking_periods.append(
            [begin_not_speaking_period, float(dfb.iloc[i]['duration'])]
        )
    return speaking_periods, not_speaking_periods


def calc_all_files_distinct_speaking_periods(dfx, df_documents):
    global_distinct_speaking_periods = []
    for file in os.listdir(AUDIO_REPOSITORY):
        filename = file[0:len(file)-12]
        speaking_periods, not_speaking_periods = distinct_speaking_periods(filename, dfx, df_documents)
        distinct_speaking_period = {
            'filename': filename,
            'speak': speaking_periods,
            'not_speak': not_speaking_periods
        }
        global_distinct_speaking_periods.append(distinct_speaking_period)
    return pd.DataFrame(global_distinct_speaking_periods)


def obtain_mfccs(df_speak):
    mfccs = []
    for wave_folder in os.listdir(WAVE_FILES):
        for wave_file in os.listdir(WAVE_FILES + '/' + wave_folder):
            file = WAVE_FILES + '/' + wave_folder + '/' + wave_file
            print(file)
            print()
            filename_df = wave_file[0 : len(wave_file) - 12]
            wr = wave.open(file, 'r')
            nchannels, sampwidth, framerate, nframes, comptype, compname =  wr.getparams()
            wr.close()
            if nchannels == 2:
                stereo = True
            else:
                stereo = False
            mfcc_x = mfcc(file, stereo=stereo, df_speak=df_speak, filename_df=filename_df)
            mfcc_x_filtre = vad_filter(mfcc_x)
            mfcc_dict = {
                'filename': filename_df,
                'mfcc': (np.asmatrix(mfcc_x_filtre)).tolist()
            }
            mfccs.append(mfcc_dict)
    df_mfcc = pd.DataFrame(mfccs)
    return df_mfcc


def delete_mfcc_empty_element(df_mfcc):
    empty_list = []
    for idx, row in df_mfcc.iterrows():
        if row['mfcc'] == []:
            empty_list.append(idx)
    df_mfcc = df_mfcc.drop([i for i in empty_list])
    return df_mfcc


def obtain_df_mean_vect(df_mfcc):
    mean_vects = []
    for idx, row in df_mfcc.iterrows():
        run = row['mfcc']
        mean_vect = {
            'filename': row['filename'],
            'mean_vect': [sum(elem) / len(elem) for elem in zip(*run)]
        }
        mean_vects.append(mean_vect)
    df_mean_vect = pd.DataFrame(mean_vects)
    return df_mean_vect