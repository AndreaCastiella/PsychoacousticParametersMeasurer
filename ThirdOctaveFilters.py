# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Funciones para filtrado en bandas de un tercio de octava

import numpy as np
import math
from scipy import signal

# Coeficientes de referencia
CoefR = np.array([[1, 2, 1, 1, -2, 1],
                  [1, 0, -1, 1, -2, 1],
                  [1, -2, 1, 1, -2, 1]])

# Coeficientes de los filtros. 28 bandas. ISO 532-1:2014
# Banda 20 Hz
Coef01 = np.array([[0, 0, 0, 0, -6.70260 * 10 ** (-4), 6.59453 * 10 ** (-4)],
                   [0, 0, 0, 0, -3.75071 * 10 ** (-4), 3.61926 * 10 ** (-4)],
                   [0, 0, 0, 0, -3.06523 * 10 ** (-4), 2.97634 * 10 ** (-4)]])
G1 = 4.30764 * 10 ** (-11)

# Banda 31,5 Hz
Coef02 = np.array([[0, 0, 0, 0, -8.47258 * 10 ** (-4), 8.30131 * 10 ** (-4)],
                   [0, 0, 0, 0, -4.76448 * 10 ** (-4), 4.55616 * 10 ** (-4)],
                   [0, 0, 0, 0, -3.88773 * 10 ** (-4), 3.74685 * 10 ** (-4)]])
G2 = 8.59340 * 10 ** (-11)

# Banda 40 Hz
Coef03 = np.array([[0, 0, 0, 0, -1.07210 * 10 ** (-3), 1.04496 * 10 ** (-3)],
                   [0, 0, 0, 0, -6.06567 * 10 ** (-4), 5.73553 * 10 ** (-4)],
                   [0, 0, 0, 0, -4.94004 * 10 ** (-4), 4.71677 * 10 ** (-4)]])
G3 = 1.71424 * 10 ** (-10)

# Banda 50 Hz
Coef04 = np.array([[0, 0, 0, 0, -1.35836 * 10 ** (-3), 1.31535 * 10 ** (-3)],
                   [0, 0, 0, 0, -7.74327 * 10 ** (-4), 7.22007 * 10 ** (-4)],
                   [0, 0, 0, 0, -6.29154 * 10 ** (-4), 5.93771 * 10 ** (-4)]])
G4 = 3.41944 * 10 ** (-10)

# Banda 63 Hz
Coef05 = np.array([[0, 0, 0, 0, -1.72380 * 10 ** (-3), 1.65564 * 10 ** (-3)],
                   [0, 0, 0, 0, -9.91780 * 10 ** (-4), 9.08866 * 10 ** (-4)],
                   [0, 0, 0, 0, -8.03529 * 10 ** (-4), 7.47455 * 10 ** (-4)]])
G5 = 6.82035 * 10 ** (-10)

# Banda 80 Hz
Coef06 = np.array([[0, 0, 0, 0, -2.19188 * 10 ** (-3), 2.08388 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.27545 * 10 ** (-3), 1.14406 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.02976 * 10 ** (-3), 9.40900 * 10 ** (-4)]])
G6 = 1.36026 * 10 ** (-9)

# Banda 100 Hz
Coef07 = np.array([[0, 0, 0, 0, -2.79386 * 10 ** (-3), 2.62274 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.64828 * 10 ** (-3), 1.44006 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.32520 * 10 ** (-3), 1.18438 * 10 ** (-3)]])
G7 = 2.71261 * 10 ** (-9)

# Banda 125 Hz
Coef08 = np.array([[0, 0, 0, 0, -3.57182 * 10 ** (-3), 3.30071 * 10 ** (-3)],
                   [0, 0, 0, 0, -2.14252 * 10 ** (-3), 1.81258 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.71397 * 10 ** (-3), 1.49082 * 10 ** (-3)]])
G8 = 5.40870 * 10 ** (-9)

# Banda 160 Hz
Coef09 = np.array([[0, 0, 0, 0, -4.58305 * 10 ** (-3), 4.15355 * 10 ** (-3)],
                   [0, 0, 0, 0, -2.80413 * 10 ** (-3), 2.28135 * 10 ** (-3)],
                   [0, 0, 0, 0, -2.23006 * 10 ** (-3), 1.87646 * 10 ** (-3)]])
G9 = 1.07826 * 10 ** (-8)

# Banda 200 Hz
Coef10 = np.array([[0, 0, 0, 0, -5.90655 * 10 ** (-3), 5.22622 * 10 ** (-3)],
                   [0, 0, 0, 0, -3.69947 * 10 ** (-3), 2.87118 * 10 ** (-3)],
                   [0, 0, 0, 0, -2.92205 * 10 ** (-3), 2.36178 * 10 ** (-3)]])
G10 = 2.14910 * 10 ** (-8)

# Banda 250 Hz
Coef11 = np.array([[0, 0, 0, 0, -7.65243 * 10 ** (-3), 6.57493 * 10 ** (-3)],
                   [0, 0, 0, 0, -4.92540 * 10 ** (-3), 3.61318 * 10 ** (-3)],
                   [0, 0, 0, 0, -3.86007 * 10 ** (-3), 2.97240 * 10 ** (-3)]])
G11 = 4.28228 * 10 ** (-8)

# Banda 315 Hz
Coef12 = np.array([[0, 0, 0, 0, -1.00023 * 10 ** (-2), 8.29610 * 10 ** (-3)],
                   [0, 0, 0, 0, -6.63788 * 10 ** (-3), 4.55999 * 10 ** (-3)],
                   [0, 0, 0, 0, -5.15982 * 10 ** (-3), 3.75306 * 10 ** (-3)]])
G12 = 8.54316 * 10 ** (-8)

# Banda 400 Hz
Coef13 = np.array([[0, 0, 0, 0, -1.31230 * 10 ** (-2), 1.04220 * 10 ** (-2)],
                   [0, 0, 0, 0, -9.02274 * 10 ** (-3), 5.73132 * 10 ** (-3)],
                   [0, 0, 0, 0, -6.94543 * 10 ** (-3), 4.71734 * 10 ** (-3)]])
G13 = 1.70009 * 10 ** (-7)

# Banda 500 Hz
Coef14 = np.array([[0, 0, 0, 0, -1.73693 * 10 ** (-2), 1.30947 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.24176 * 10 ** (-2), 7.20526 * 10 ** (-3)],
                   [0, 0, 0, 0, -9.46002 * 10 ** (-3), 5.93145 * 10 ** (-3)]])
G14 = 3.38215 * 10 ** (-7)

# Banda 630 Hz
Coef15 = np.array([[0, 0, 0, 0, -2.31934 * 10 ** (-2), 1.64308 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.73009 * 10 ** (-2), 9.04761 * 10 ** (-3)],
                   [0, 0, 0, 0, -1.30358 * 10 ** (-2), 7.44926 * 10 ** (-3)]])
G15 = 6.71990 * 10 ** (-7)

# Banda 800 Hz
Coef16 = np.array([[0, 0, 0, 0, -3.13292 * 10 ** (-2), 2.06370 * 10 ** (-2)],
                   [0, 0, 0, 0, -2.44342 * 10 ** (-2), 1.13731 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.82108 * 10 ** (-2), 9.36778 * 10 ** (-3)]])
G16 = 1.33531 * 10 ** (-6)

# Banda 1 kHz
Coef17 = np.array([[0, 0, 0, 0, -4.28261 * 10 ** (-2), 2.59325 * 10 ** (-2)],
                   [0, 0, 0, 0, -3.49619 * 10 ** (-2), 1.43046 * 10 ** (-2)],
                   [0, 0, 0, 0, -2.57855 * 10 ** (-2), 1.17912 * 10 ** (-2)]])
G17 = 2.65172 * 10 ** (-6)

# Banda 1,25 kHz
Coef18 = np.array([[0, 0, 0, 0, -5.91733 * 10 ** (-2), 3.25054 * 10 ** (-2)],
                   [0, 0, 0, 0, -5.06072 * 10 ** (-2), 1.79513 * 10 ** (-2)],
                   [0, 0, 0, 0, -3.69401 * 10 ** (-2), 1.48094 * 10 ** (-2)]])
G18 = 5.25477 * 10 ** (-6)

# Banda 1,6 kHz
Coef19 = np.array([[0, 0, 0, 0, -8.26348 * 10 ** (-2), 4.05894 * 10 ** (-2)],
                   [0, 0, 0, 0, -7.40348 * 10 ** (-2), 2.24476 * 10 ** (-2)],
                   [0, 0, 0, 0, -5.34977 * 10 ** (-2), 1.85371 * 10 ** (-2)]])
G19 = 1.03780 * 10 ** (-5)

# Banda 2 kHz
Coef20 = np.array([[0, 0, 0, 0, -1.17018 * 10 ** (-1), 5.08116 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.09516 * 10 ** (-1), 2.81387 * 10 ** (-2)],
                   [0, 0, 0, 0, -7.85097 * 10 ** (-2), 2.32872 * 10 ** (-2)]])
G20 = 2.04870 * 10 ** (-5)

# Banda 2,5 kHz
Coef21 = np.array([[0, 0, 0, 0, -1.67714 * 10 ** (-1), 6.37872 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.63378 * 10 ** (-1), 3.53729 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.16419 * 10 ** (-1), 2.93723 * 10 ** (-2)]])
G21 = 4.05198 * 10 ** (-5)

# Banda 3,15 kHz
Coef22 = np.array([[0, 0, 0, 0, -2.42528 * 10 ** (-1), 7.98576 * 10 ** (-2)],
                   [0, 0, 0, 0, -2.45161 * 10 ** (-1), 4.43370 * 10 ** (-2)],
                   [0, 0, 0, 0, -1.73972 * 10 ** (-1), 3.70015 * 10 ** (-2)]])
G22 = 7.97914 * 10 ** (-5)

# Banda 4 kHz
Coef23 = np.array([[0, 0, 0, 0, -3.53142 * 10 ** (-1), 9.96330 * 10 ** (-2)],
                   [0, 0, 0, 0, -3.69163 * 10 ** (-1), 5.53535 * 10 ** (-2)],
                   [0, 0, 0, 0, -2.61399 * 10 ** (-1), 4.65428 * 10 ** (-2)]])
G23 = 1.56511 * 10 ** (-4)

# Banda 5 kHz
Coef24 = np.array([[0, 0, 0, 0, -5.16316 * 10 ** (-1), 1.24177 * 10 ** (-1)],
                   [0, 0, 0, 0, -5.55473 * 10 ** (-1), 6.89403 * 10 ** (-2)],
                   [0, 0, 0, 0, -3.93998 * 10 ** (-1), 5.86715 * 10 ** (-2)]])
G24 = 3.04954 * 10 ** (-4)

# Banda 6,3 kHz
Coef25 = np.array([[0, 0, 0, 0, -7.56635 * 10 ** (-1), 1.55023 * 10 ** (-1)],
                   [0, 0, 0, 0, -8.34281 * 10 ** (-1), 8.58123 * 10 ** (-2)],
                   [0, 0, 0, 0, -5.94547 * 10 ** (-1), 7.43960 * 10 ** (-2)]])
G25 = 5.99157 * 10 ** (-4)

# Banda 8 kHz
Coef26 = np.array([[0, 0, 0, 0, -1.10165 * 10 ** (+0), 1.91713 * 10 ** (-1)],
                   [0, 0, 0, 0, -1.23939 * 10 ** (+0), 1.05243 * 10 ** (-1)],
                   [0, 0, 0, 0, -8.91666 * 10 ** (-1), 9.40354 * 10 ** (-2)]])
G26 = 1.16544 * 10 ** (-3)

# Banda 10 kHz
Coef27 = np.array([[0, 0, 0, 0, -1.58477 * 10 ** (+0), 2.39049 * 10 ** (-1)],
                   [0, 0, 0, 0, -1.80505 * 10 ** (+0), 1.28794 * 10 ** (-1)],
                   [0, 0, 0, 0, -1.32500 * 10 ** (+0), 1.21333 * 10 ** (-1)]])
G27 = 2.27488 * 10 ** (-3)

# Banda 12,5 kHz
Coef28 = np.array([[0, 0, 0, 0, -2.50630 * 10 ** (+0), 1.42308 * 10 ** (-1)],
                   [0, 0, 0, 0, -2.19464 * 10 ** (+0), 2.76470 * 10 ** (-1)],
                   [0, 0, 0, 0, -1.90231 * 10 ** (+0), 1.47304 * 10 ** (-1)]])
G28 = 3.91006 * 10 ** (-3)

# Todas las bandas
CoeficientesArray = np.stack((Coef01, Coef02, Coef03, Coef04, Coef05, Coef06, Coef07, Coef08, Coef09,
                              Coef10, Coef11, Coef12, Coef13, Coef14, Coef15, Coef16, Coef17, Coef18,
                              Coef19, Coef20, Coef21, Coef22, Coef23, Coef24, Coef25, Coef26, Coef27,
                              Coef28))

# Ganancia para los tres filtrados
GananciaFilt = np.array([(G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20,
                          G21, G22, G23, G24, G25, G26, G27, G28),
                         (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
                         (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)])

# Frecuencias centrales
Frecuencias = np.array([25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                        2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

constantes = {
    'N_LEVEL_BANDS': 28,
    'SR_LEVEL': 2000,
    'I_REF': 4 * 10 ** -10,
    'TINY_VALUE': 10 ** -12,
    'N_FILTER_STAGES': 3,
    'N_FILTER_COEFS': 6
}


# Filtrado en bandas de tercio de octava sonidos estacionarios
def ThirdOctaveBandFilter(frame, CHUNK=4800):
    # Filtrado
    Coeficientes = np.zeros(constantes['N_FILTER_COEFS'])
    ThirdOctaveBands = np.zeros((CHUNK, constantes['N_LEVEL_BANDS']))
    SignalOut = 0

    for idxFB in range(constantes['N_LEVEL_BANDS']):  # Por cada banda
        SignalIn = frame.copy()
        for idxFS in range(constantes['N_FILTER_STAGES']):  # Filtrados por banda (3)
            for idxC in range(constantes['N_FILTER_COEFS']):
                Coeficientes[idxC] = CoefR[idxFS, idxC] - CoeficientesArray[idxFB, idxFS, idxC]

            Ganancia = GananciaFilt[idxFS][idxFB]

            SignalOut = filter_2ndOrder(SignalIn,
                                        Coeficientes, 
                                        CHUNK, 
                                        Ganancia)
            SignalIn = SignalOut

            if idxFS == 2:  # En el ultimo
                ThirdOctaveBands[:, idxFB] = SignalOut[:]
    return ThirdOctaveBands


# Cálculo nivel SPL en bandas de tercio de octava
def ThirdOctaveSPL(ThirdOctaveBands, CHUNK=4800, RATE=48000, TimeSkip=0, TimeVarying=False):
    # Valor pequeño para evitar que haya un 0 si no se capta señal
    ThirdOctaveLevel = np.zeros(np.shape(ThirdOctaveBands)[1])

    if not TimeVarying:
        # Suavizado
        NumSkip = np.floor(TimeSkip * RATE)

        if NumSkip >= CHUNK:
            print('Señal demasiado corta')
        Out = 0
        for idxFB in range(np.shape(ThirdOctaveBands)[1]):  # Numero de bandas. De 0 a 27
            SignalBand = ThirdOctaveBands[:, idxFB]
            for Time in range(int(NumSkip), CHUNK):
                Out = SignalBand[Time] ** 2 + Out
            Out = Out / (CHUNK - NumSkip)
            ThirdOctaveLevel[idxFB] = 10 * math.log10((Out + constantes['TINY_VALUE']) / constantes['I_REF'])
    return ThirdOctaveLevel


# Filtrado en bandas de tercio de octava sonidos variantes en el tiempo y cálculo niveles
def ThirdOctaveLevelTime(frame, RATE=48000, CHUNK=4800):
    constantes['DEC_FACTOR'] = int(RATE / constantes['SR_LEVEL']) # para calculo mas rapido
    Coeficientes = np.zeros(constantes['N_FILTER_COEFS'])

    n_time = len(frame[::constantes['DEC_FACTOR']])  # Longitud frame cogiendo los valores con saltos de DEC_FACTOR
    time_axis = np.linspace(0, CHUNK / RATE, num=n_time)  # linspace(comienzo, fin, numero de valores equiespaciados entre ellos)
    ThirdOctaveLevel = np.zeros((constantes['N_LEVEL_BANDS'], n_time))

    for idxFB in range(constantes['N_LEVEL_BANDS']): 
        SignalIn = frame.copy()
        for idxFS in range(constantes['N_FILTER_STAGES']):  # Filtrados por banda (3)
            for idxC in range(constantes['N_FILTER_COEFS']):
                Coeficientes[idxC] = CoefR[idxFS, idxC] - CoeficientesArray[idxFB, idxFS, idxC]
            
            Gain = GananciaFilt[idxFS][idxFB]
            SignalOut = filter_2ndOrder(SignalIn, 
                                        Coeficientes, 
                                        CHUNK, 
                                        Gain)
            SignalIn = SignalOut

        # Calculo frecuencia central del filtro
        centerFreq = np.power(10, (idxFB - 16) / 10, dtype=float) * 1000
        # Elevar al cuadrado y suavizar
        filtrada = f_square_and_smooth(SignalOut, centerFreq, RATE, CHUNK)
        
        # Calculo SPL y decimacion
        ThirdOctaveLevel[idxFB][:] = 10 * np.log10((np.array((filtrada[::constantes['DEC_FACTOR']])).flatten() + constantes['TINY_VALUE']) / constantes['I_REF'])
    return ThirdOctaveLevel, Frecuencias, time_axis


# Elevar al cuadrado y suavizar
def f_square_and_smooth(frame, centerFreq, RATE=48000, CHUNK=4800):
    Input = frame.copy()
    # Constante de tiempo dependiente de la frecuencia
    if centerFreq <= 1000:
        Tau = 2 / (3.0 * centerFreq)
    else:
        Tau = 2 / (3.0 * 1000.0)
    # Cuadrado
    Output = Input**2
    # 3 filtros suavizantes paso bajo
    a1 = np.exp(-1 / (RATE * Tau))
    b0 = 1 -a1
    for i in range(3):
        Output = filter_lowpass_1stOrder(Output, 
                                         Tau, 
                                         RATE, 
                                         CHUNK)
    return Output


# Filtro paso bajo primer orden
def filter_lowpass_1stOrder(frame, Tau, RATE=48000, CHUNK=4800):
    Y = 0
    Input = frame.copy()
    Output = np.zeros(CHUNK)

    A1 = np.exp(-1 / (RATE * Tau))
    B0 = 1 - A1

    for i in range(CHUNK):
        Output[i] = B0 * Input[i] + A1 * Y
        Y = Output[i]
    return Output


# Filtrado segundo orden
def filter_2ndOrder(frame, Coef, CHUNK=4800, Gain=1):
    Input = frame.copy()
    Output = np.zeros(CHUNK)
    Wn1 = 0
    Wn2 = 0

    for i in range(CHUNK):
        Wn0 = float(Input[i]) * float(Gain) - float(Coef[4]) * Wn1 - float(Coef[5]) * Wn2
        Output[i] = float(Coef[0]) * Wn0 + float(Coef[1]) * Wn1 + float(Coef[2]) * Wn2
        Wn2 = Wn1
        Wn1 = Wn0
    return Output