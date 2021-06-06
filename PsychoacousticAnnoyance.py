# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Psychoacoustic Annoyance Zwicker y Fastl "Psychoacoustics: Facts and Models"
import numpy as np

# Cálculo PA
def psychoacousticAnnoyance(L, S, FS, R):
    N5 = calcPercentile(L)
    if N5 == 0:
        N5 = 1e-8
    ws = 0
    if S > 1.75:
        ws = (S-1.75)*0.25*np.log10(N5+10)     
    wfr = (2.18/N5**0.4)*(0.4*FS + 0.6*R)
    PA = N5*(1 + np.sqrt(ws**2+wfr**2))
    return PA


# Cálculo percentil
def calcPercentile(x, N=5):
    xsort = np.sort(x)
    l = len(x)
    pos = np.ceil(0.01*N*l)
    return xsort[int(pos)-1]