# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Fluctuation Strength Zwicker y Fastl "Psychoacoustics: Facts and Models"

from loudness_ISO532 import sone2phon
import numpy as np
from scipy import signal

# Cálculo de FS
def acousticFluctuation(specificLoudness, fmod=4):
    specificLoudnessdiff = np.zeros(len(specificLoudness))
    for i in range(len(specificLoudness)):
        if i == 0:
            specificLoudnessdiff[i] = specificLoudness[i]
        else:
            specificLoudnessdiff[i] = abs(specificLoudness[i] - specificLoudness[i-1])
    F = (0.008*sum(0.1*specificLoudnessdiff))/((fmod/4)+(fmod/4))
    return F


# Detección de la frecuencia de modulación
def fmoddetection(specificLoudness, fmin=.2, fmax=64):
    if len(specificLoudness.shape) == 1:
        specificLoudness = np.reshape(specificLoudness, (1, len(specificLoudness)))
    else:
        specificLoudness = np.transpose(specificLoudness)
    phon = np.zeros((specificLoudness.shape[0], specificLoudness.shape[1], 1))
    for i, ms in enumerate(specificLoudness):
        for j, bark in enumerate(ms):
            phon[i, j, 0] = sone2phon(bark)
    FSpec = 1/0.002

    # Reagrupar saluda en bandas de 24 o 47 Bark
    phon1024a = np.reshape(phon,  newshape = (24, phon.shape[0], 10))
    # Overlap
    phon1024b = np.reshape(phon[:, 5:235],  newshape = ( 23, phon.shape[0], 10)) # Deja fuera los primeros 5 y los últimos 5
    h = np.hamming(10)
    phonB = np.zeros((phon.shape[0], 47))
    phonBtempa = np.sum((phon1024a * h), 2)
    phonB[:,0:47:2] = np.transpose(phonBtempa)
    phonBtempb = np.sum((phon1024b * h), 2)
    phonB[:,1:46:2] = np.transpose(phonBtempb)
    
    phonBm = phonB - (np.mean(phonB, 0))    # Eliminar media
    phonBm = np.maximum(0, phonBm)  # Eliminar parte negativa
    pbfstd = 5*(np.std(phonB, 0, ddof=1))   # Recortar amplitudes extremas
    phonBm = np.minimum(phonBm, pbfstd)
    phonBf = hpfilt(phonBm) # Eliminar bajas frecuencias
    pbfstd2 = np.std(phonBf, 0, ddof=1) # Recortar amplitudes extremas
    phonBf = np.minimum(phonBf, pbfstd2)
    phonBf = np.maximum(phonBf, -pbfstd2)
    NP = np.maximum(8192, 2**(next_power_of_2(specificLoudness.shape[0])))
    if not NP <= np.maximum(8192, 2*specificLoudness.shape[0]):
        print('Error')
        return None
    fmin = np.int(np.floor(NP*fmin/FSpec))    # Frecuncia de detección mínima
    fmax = np.int(np.ceil(NP*fmax/FSpec))    # Frecuencia de detección máxima
    X = np.sum(np.abs(np.fft.fft(phonBf,np.int(NP), axis=0)), 1)
    if not X.shape[0] <= NP:
        print('Error')
        return None
    nf = fmax-fmin+1
    t = np.arange(fmin,fmax+1)
    b = np.matmul(np.linalg.pinv(np.stack((np.ones(nf), t), axis=1)), np.log(np.maximum(X[fmin-1:fmax], np.finfo(float).tiny))) # Evitar log(0)
    XR = np.exp(b[1]*t+b[0])
    idx = np.argmax(X[fmin-1:fmax]- XR)
    idx = idx + fmin - 1
    cf = (idx)*FSpec/(NP-1)
    if idx > 4:                                 
        cf = refineCF(cf, X[idx-3:idx+2], NP, FSpec)
    return cf


# Cálculo potencia de dos inmediata
def next_power_of_2(x):
    return 1 if x == 0 else np.log2(2**(x - 1).bit_length())


# Filtro paso alto
def hpfilt(x):
    sos = np.array([
        [.998195566288485491845960950741, -1.996391132576970983691921901482, .998195566288485491845960950741, 1, -1.994897079594937228108619819977, .994904703139997681482498137484],
        [.998195566288485491845960950741, -1.996391132576970983691921901482, .998195566288485491845960950741, 1, -1.997878669564554510174048118643, .997886304503829646428414434922]])
    fn = 10**4
    xL = np.vstack((x, np.zeros((fn, x.shape[1]))))
    xLf = np.flipud(signal.sosfilt(sos, np.flipud(signal.sosfilt(sos, xL))))
    return xLf[0:x.shape[0], :]


# refinar frecuencia detectada
def refineCF(cf, p, NP, FSpec):
    rd_denom = p[0]+p[4]-16*(p[1]+p[3])+30*p[2]
    if not rd_denom == 0:
        rd = (p[0]-p[4]+8*(p[3]-p[1])) / rd_denom
        if np.abs(rd) < .5:
            cf = cf + rd*FSpec/(NP-1)
        else:
            p = p[1:4]
            if p[0]-2*p[1]+p[2] < 0:
                rd_denom = 2*(p[0]+p[2]-2*p[1])
                if not rd_denom == 0:
                    rd = (3*p[0]-4*p[1]+p[2])
                    if np.abs(rd) < .5:
                        cf = cf + rd*FSpec/(NP-1)
    return cf