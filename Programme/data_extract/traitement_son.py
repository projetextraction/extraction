# -*- coding: utf-8 -*-

import os
from bs4 import BeautifulSoup as Soup


METADATA_FILE = os.path.join(os.getcwd(), "Data\\Txt\\DEV_M2SID_METADATA")
TRANS_FILE = os.path.join(os.getcwd(), "Data\\Audio\\DEV_M2SID_LIMSI_ASR")

def extract_mots(mots, speech_seg, pk_mot, pk_segment):
    words_list = speech_seg.find_all('Word')
    for word in words_list:
        mot = {
            'pk': pk_mot,
            'segment': pk_segment,
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
            'pk': pk_segment,
            'speaker': speech_seg.get('spkid'),
            'stime': speech_seg.get('stime'),
            'etime': speech_seg.get('etime'),
            'lang': speech_seg.get('lang')
        }
        segment['speaker'] = speech_seg.get('spkid')
        ####
        idx = sp_kids.index(segment['speaker'])
        segment['locuteur'] = pks_locuteur[idx]
        ####
        mots, pk_mot = extract_mots(mots, speech_seg, pk_mot, segment['pk'])
        if segment != {}:
            segments.append(segment)
            pk_segment += 1
    return segments, mots, pk_segment, pk_mot


# vecteurs locuteurs: gender, temps_parole, nombre de speechsegments,
#                     temps moyen de paroles


def extract_locuteurs(locuteurs, segments, mots, sp, pk_document, pk_locuteur, pk_segment, pk_mot):
    speakers = sp.find_all('Speaker')
    pks_locuteur , spkids = [], []
    for speaker in speakers:
        locuteur = {
            'pk': pk_locuteur,
            'gender': speaker.get('gender'),
            'speaker': speaker.get('spkid'),
            'temps_parole': speaker.get('dur'),
            'document': pk_document
        }
        pks_locuteur.append(pk_locuteur)
        spkids.append(locuteur['speaker'])
        pk_locuteur += 1
        locuteurs.append(locuteur)
    segments, mots, pk_segment, pk_mots =  extract_segments(
        segments, mots, sp, pks_locuteur, spkids, pk_segment, pk_mot
    )
    return locuteurs, segments, mots, pk_locuteur, pk_segment, pk_mot


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
        document = {
            'pk': pk_document,
            'nom': trans_file
        }
        locuteurs, segments, mots, pk_locuteur, pk_segment, pk_mots =  extract_locuteurs(
            locuteurs, segments, mots, sp, pk_document, pk_locuteur, pk_segment, pk_mot
        )
        pk_document += 1
        documents.append(document)
    return documents, locuteurs, segments, mots


documents, locuteurs, segments, mots = extract_documents()