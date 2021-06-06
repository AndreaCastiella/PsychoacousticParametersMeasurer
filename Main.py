# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Main

import math
import numpy as np
import pyaudio
import loudness_ISO532
import sys
import ThirdOctaveFilters as TOF
import Sharpness
from FiltroPonderacionA import filtroPonderacionA
from Roughness import acousticRoughness
from FluctuationStrength import acousticFluctuation, fmoddetection
from PsychoacousticAnnoyance import psychoacousticAnnoyance

# Constantes e inicialización pyaudio
CHUNK = 4800  # Tamaño en muestras almacenadas en cada array
RATE = 48000  # Muestras por segundo
TimeVarying = False
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, 
					rate=RATE, 
					channels=1, 
					input=True, 
					input_device_index=2,
					frames_per_buffer=CHUNK)  
					# Input Device = 0 si ordenador, 1 si cascos, 2 si Raspberry

# Funciones principales
def mainEstacionario(data, RATE, CHUNK):
    # Filtro tercio de octava
    ThirdOctave = TOF.ThirdOctaveBandFilter(frame=data, CHUNK=CHUNK)
    ThirdOctaveSPL = TOF.ThirdOctaveSPL(ThirdOctaveBands=ThirdOctave, 
                                        CHUNK=CHUNK, 
                                        RATE=RATE, 
                                        TimeSkip=0)
    # Loudness
    loudness, specLoudness, _, _ = loudness_ISO532.loudness_ISO532(ThirdOctaveSPL,
                                                                   SoundFieldDiffuse=0)
    loudnessPhon = loudness_ISO532.sone2phon(loudness)
    print('Loudness total en sonios: ', round(loudness, 1), ' sonios')
    print('Loudness total en fonos', round(loudnessPhon, 2), ' fonos')

    # Cálculo sharpness
    sharpnessZwicker, sharpnessVonBismarck, sharpnessAures = Sharpness.calc_sharpness(loudness, 
                                                                                      specLoudness)
    print('Sharpness Zwicker: ', sharpnessZwicker, ' acum')
    print('Sharpness VB: ', sharpnessVonBismarck, ' acum')
    print('Sharpness Aures: ', sharpnessAures, ' acum')

    
    specLoudness2 = np.stack((specLoudness, specLoudness), axis=1)
    fmodFS = fmoddetection(specLoudness2, fmin=.2, fmax=64)
    #Fluctuation strength
    FS = acousticFluctuation(specLoudness, fmodFS)
    fmodR = fmoddetection(specLoudness2, fmin=40, fmax=150)
    # Roughness
    R = acousticRoughness(specLoudness, fmodR)
    # Psychoacoustic Annoyance
    PA = psychoacousticAnnoyance(specLoudness, sharpnessZwicker, FS, R)

    print('Fluctuation strength: ', round(FS, 2), ' vacil')
    print('Roughness: ', round(R, 2), ' asper')
    print('Psychoacoustic annoyance: ', round(PA, 2))

def mainVarianteTiempo(data, RATE, CHUNK):
    # Filtrado tercio de octava
    ThirdOctaveLevelTime, _, _ = TOF.ThirdOctaveLevelTime(data)
    # Loudness
    loudness, specLoudness = loudness_ISO532.loudness_ISO532_time(ThirdOctaveLevelTime, 
                                                                  SoundFieldDiffuse=0)

    loudnessPhon = []
    for loundness_i in loudness:
        loudnessPhon.append(loudness_ISO532.sone2phon(loundness_i))
    
    loudnessPhon = [round(num, 1) for num in loudnessPhon]

    print('Loudness total en sonios: ', loudness, ' sonios')
    print('Loudness total en fonos', loudnessPhon, ' fonos')

    sharpnessZwicker = []
    sharpnessVonBismarck = []
    sharpnessAures = []

    for i in range(len(loudness)):
        sharpnessZwickerTemp, sharpnessVonBismarckTemp, sharpnessAuresTemp = Sharpness.calc_sharpness(loudness[i], 
                                                                                                      specLoudness[:, i])
        sharpnessZwicker.append(sharpnessZwickerTemp)
        sharpnessVonBismarck.append(sharpnessVonBismarckTemp)
        sharpnessAures.append(sharpnessAuresTemp)

    print('Sharpness Zwicker: ', sharpnessZwicker)
    print('Sharpness VB: ', sharpnessVonBismarck)
    print('Sharpness Aures: ', sharpnessAures)

    # Detección frecuencia moduladora
    fmodFS = fmoddetection(specLoudness, fmin=.2, fmax=64)
    fmodR = fmoddetection(specLoudness, fmin=40, fmax=150)

    FS = []
    R = []
    PA = []

    for i in range(len(loudness)):
        #Fluctuation strength
        FS.append(acousticFluctuation(specLoudness[:, i], fmodFS))
        # Roughness
        R.append(acousticFluctuation(specLoudness[:, i], fmodR))
        # Psychoacoustic Annoyance
        PA.append(psychoacousticAnnoyance(specLoudness[:, i], sharpnessZwicker[i], FS[i], R[i]))

    FS = [round(num, 4) for num in FS]
    R = [round(num, 4) for num in R]
    PA = [round(num, 4) for num in PA]

    print('Fluctuation strength: ', FS, ' vacil')
    print('Roughness: ', R, ' asper')
    print('Psychoacoustic annoyance: ', PA)

def captarAudio(corr_factor = 4.73):
    # Almacenar audio en array
    data = np.frombuffer(stream.read(CHUNK, 
                         exception_on_overflow=False), 
                         dtype=np.float32)
    # Factor de corrección del dispositivo que está siendo utilizado para captar la señal. 94 dB a 1 kHz 1 Pa de referencia.
    data = data * corr_factor
    return data

def calcSPL_SPLA(data):
    # Cálculo del valor rms de la señal captada (CHUNK)
    rms_corr = np.sqrt(np.mean(np.absolute(data.astype(float)) ** 2))
    # Cálculo del valor SPL
    SPL = 20 * math.log(rms_corr / (20 * (10 ** -6)), 10)
    print("Z-weighted", round(SPL,1))

    # Fitlrado ponderación A
    pond_A = filtroPonderacionA(data, RATE)
    # Cálculo del valor rms de la señal ponderada A y corrección
    rms_A = np.sqrt(np.mean(np.absolute(pond_A.astype(float)) ** 2))
    rms_A_corr = rms_A
    # Cálculo del valor SPL ponderado A
    SPL_A = 20 * math.log(rms_A_corr / (20 * (10 ** -6)), 10)
    print("A-weighted", round(SPL_A,1))

try:
    # Bucle para captar audio de forma continua
    while True:  
        # Captación audio
        data = captarAudio()

        # Cálculo SPL y SPL(A)
        calcSPL_SPLA(data)

        # Estacionario
        if not TimeVarying:
            mainEstacionario(data, RATE, CHUNK)

        # Variante en el tiempo
        if TimeVarying:
            mainVarianteTiempo(data, RATE, CHUNK)

# Salir del bucle con ctr+c
except KeyboardInterrupt:
    # Cierre stream
    stream.stop_stream()
    stream.close()
    p.terminate
    print('Interrupción de teclado. Finalizando programa.')
    sys.exit()


