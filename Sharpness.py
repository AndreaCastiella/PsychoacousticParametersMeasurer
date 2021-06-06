# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Sharpness DIN 45692

import numpy as np

def calc_sharpness(loudness, specificLoudness):
	# Evitar división entre 0
    if loudness == 0:
        loudness = 1e-8
    n = len(specificLoudness)

    # Zwicker method
    gz_Z = np.ones(n)
    for z in range(158, n):
        gz_Z[z] = 0.15*np.e**(0.42 * ((z+1)/10-15.8))+0.85
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += specificLoudness[i] * gz_Z[i] * round(z, 1) * 0.1
    sharpnessZwicker = 0.11 * suma / loudness

    # Von Bismarck method
    gz_VB = np.ones(n)
    for z in range(150, n):
        gz_VB[z] = 0.2 * np.e ** (0.308 * ((z+1)/10 - 15)) + 0.8
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += specificLoudness[i] * gz_VB[i] * round(z, 1) * 0.1
    sharpnessVonBismarck = 0.11 * suma / loudness

    # Aures method
    gz_A = np.ones(n)
    for z in range(0, n):
    	zi = np.float((np.float(z+1))/np.float(10))
        gz_A[z] = (0.078*np.e**(0.171*(zi))/(zi))*(loudness/np.log(0.05*loudness+1))
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += specificLoudness[i] * gz_A[i] * round(z, 1) * 0.1
    sharpnessAures = 0.11 * suma / loudness

    return round(sharpnessZwicker, 2), round(sharpnessVonBismarck, 2), round(sharpnessAures, 2)
