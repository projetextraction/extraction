#!/usr/bin/python
# coding: utf-8
import numpy as np
import numpy.linalg as la
from math import *
from numpy import zeros
from scipy.io import wavfile


def hamming2(n):
    """ Define Hamming Window and use it to do short 1me fourier transform """
    return 0.54 - 0.46* np.cos(2*pi/n * np.arange(n))

def mel(nFilters, FFTLen, sampRate): 
    halfFFTLen = int(floor(FFTLen/2)) 
    M = zeros((nFilters, halfFFTLen)) 
    lowFreq = 20 # Hz
    highFreq = 8000 # Hz
    melLowFreq = 1125*np.log(1+lowFreq/700.0)
    melHighFreq = 1125*np.log(1+highFreq/700.0)
    melStep = int(floor((melHighFreq - melLowFreq)/nFilters)) 
    melLow2High = np.arange(melLowFreq, melHighFreq, melStep) 
    #melLow2High = 1125*np.log(1+np.arange(lowFreq, highFreq)/700.0)
    HzLow2High = 700*(np.exp(melLow2High/1125)-1) 
    HzLow2HighNorm = np.floor(FFTLen*HzLow2High/sampRate)

    # form the triangular filters 
    for filt in range(nFilters):
        xStart1 = HzLow2HighNorm[filt] 
        xStop1 = HzLow2HighNorm[filt+1] 
        yStep1 = 1/(xStop1-xStart1) 
        M[filt, int(xStart1)] = 0.0;
        for x in np.arange(xStart1+1, xStop1): 
            M[filt, int(x)] = M[filt, int(x)-1] + yStep1
    for filt in range(nFilters-1):
        xStart2 = HzLow2HighNorm[filt+1]
        xStop2 = HzLow2HighNorm[filt+2] 
        yStep2 = -1.0/(xStop2-xStart2)
        M[filt, int(xStart2)] = 1.0;
        for x in np.arange(xStart2+1, xStop2):
            M[filt, int(x)] = M[filt, int(x)-1] + yStep2
    return melLow2High, HzLow2High, HzLow2HighNorm, M

def dctmtx(n):
    #DCT-II matrix
    x,y = np.meshgrid(range(n), range(n))
    D = np.sqrt(2.0/n) * np.cos(pi * (2*x+1) * y / (2*n)) 
    D[0] /= np.sqrt(2)
    return D


def extract_info_wave_file(freq, son, df_speak, filename):
    '''Extrait les moments de paroles et de non paroles'''
    non_speak, speak = [], []
    df = df_speak.loc[df_speak['filename'] == filename]
    # Normally, df is a dataframe with a single element
    spk = df.iloc[0]['speak']
    nspk = df.iloc[0]['not_speak']
    # Normally, len(nspk) == len(spk) + 1
    for i in range(len(nspk)):
        if i != len(nspk) - 1:
            extrait = son[int(nspk[i][0]*freq) : int(nspk[i][1]*freq)]
            non_speak.append(extrait)
            extrait = son[int(spk[i][0]*freq) : int(spk[i][1]*freq)]
            speak.append(extrait)
        else:
            extrait = son[int(nspk[i][0]*freq) : len(son)]
            non_speak.append(extrait)
    return non_speak, speak


def mfcc(filename, stereo=False, df_speak=None, filename_df=None):
    ''' MFCC feature extractor
        From: Yangshun Tay https://gist.github.com/yangshun/9183815'''
    sampleRate, signal = wavfile.read(filename)
    #
    if stereo:
        signal = signal.sum(axis=1) / 2
    non_speak, speak = extract_info_wave_file(sampleRate, signal, df_speak, filename_df)
    signal = np.concatenate([non_speak[i] for i in range(len(non_speak))])
    #
    FrameDuration = 0.020 # 20 milliseconds window
    FrameLen = int(FrameDuration * sampleRate) # number of points in one window
    FrameShift = FrameLen / 2 # overlapping window
    FFTLen = 2048
    win = hamming2(FrameLen)
    
    # spectrogram
    lenSig = len(signal)
    nframes = int((lenSig-FrameLen)/FrameShift)
    nFilters = 40
    mfccCoefs = 13
    preEmphFactor = 0.95 #array to hold spectrogram
    powSpec2D = np.zeros((FFTLen,nframes))
    mfcc2D = np.zeros((mfccCoefs,nframes))
    mfcc2DSpec = np.zeros((nFilters,nframes))
    mfcc2DPow = np.zeros((nFilters,nframes))
    melLow2High, HzLow2High, HzLow2HighNorm, M = mel(nFilters, FFTLen, sampleRate)
    D = dctmtx(nFilters)[1:mfccCoefs+1]
    invD = la.inv(dctmtx(nFilters))[:,1:mfccCoefs+1]
    minPowSpec = 1e-50

    for fr in range(0, nframes):
        start = int(fr*FrameShift)
        currentFrame = signal[start:start+FrameLen]
        #--- pre-emphasis filtering
        #currentFrame[1:] -= currentFrame[:-1] * preEmphFactor
    
        currentFrame1 = currentFrame * win
    
        #--- pre-emphasis filtering
        currentFrame1[1:] -= currentFrame1[:-1] * preEmphFactor
    
        #--- fourier transform using numpy
        fftCurrentFrame = np.fft.fft(currentFrame1, FFTLen)
        fftCurrentFrame[abs(fftCurrentFrame) < minPowSpec] = minPowSpec

        shiftedFFT = np.fft.fftshift(fftCurrentFrame)
        powSpec = 20*np.log(np.abs(shiftedFFT))

        #--- store current frame's power spectrum
        powSpec2D[:,fr] = powSpec
        #mfcc2D[:, fr] = np.log(np.dot(M, np.abs(shiftedFFT)**2))
        mfcc2DPow[:, fr] = np.log(np.dot(M, np.abs(fftCurrentFrame[0:int(FFTLen/2)])**2))
        mfcc2D[:, fr] = np.dot(D, mfcc2DPow[:,fr])

    # normalize the MFCC coefficients
    meanFeat = np.mean(mfcc2D, axis=0)
    sigmaFeat = np.std(mfcc2D, axis=0)
    mfcc2D = (mfcc2D - meanFeat)/sigmaFeat

    for fr in range(0, nframes-1):
        mfcc2DSpec[:, fr] = np.dot(invD, mfcc2D[:,fr].T)

    return mfcc2D.T


def vad_filter(matriceMFCC):
    ''' Filtrage de la matrice des MFCC pour ne conserver que les lignes
    qui correspondent a de la parole'''
    # --- A FAIRE ----
    C0_glob = matriceMFCC[:,-1].mean()
    seuil = C0_glob * 70/100
    matriceMFCCaux = matriceMFCC[~(matriceMFCC[:,-1] < seuil)]
    matriceMFCCfiltre = matriceMFCCaux[:,0:12]
    return matriceMFCCfiltre