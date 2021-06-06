# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Loudness ISO 532-1:2014
# Método para sonidos estacionarios y variables con el tiempo

import numpy as np

constantes = {
    'SR_LEVEL': 2000,
    'N_BARK_BANDS': 240,
    'N_LCB_BANDS': 11,
    'N_CORE_LOUDN': 21,
    'N_RAP_RANGES': 8,
    'N_RNS_RANGES': 18,
    'N_CB_RANGES': 8,
    'NL_ITER': 24,
    'LP_ITER': 24,
    'SOUNDFIELDDIFFUSE': 1,
    'SOUNDFILEDFREE': 0,
    'TSHORT': 0.005,
    'TLONG': 0.015,
    'TVAR': 0.075,
    'DEC_FACTOR': 4 # Pasar de 0.5 ms a 2 ms
}

variablesBark = {
    'N_BARK_BANDS': 24
}


# Corrección para bajas frecuencias
def f_corr_third_octave_intensities(ThirdOctaveLevel):
    ThirdOctaveIntens = np.zeros(11)
    # Por número de LCB BANDS de 25 a 250 Hz
    for idxIntens in range(constantes['N_LCB_BANDS']):
        # Ponderación de los niveles de un tercio de octava para bandas entre 25 y 250 Hz
        DLL = np.array([[-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
                        [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
                        [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
                        [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
                        [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
                        [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
                        [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
                        [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0]])
        # Rangos de nivel para ponderación de los niveles
        RAP = np.array([45, 55, 65, 71, 80, 90, 100, 120])

        # Corrección primeras 11 bandas 
        idxLevelRange = 0
        while (ThirdOctaveLevel[idxIntens] > RAP[idxLevelRange] - DLL[idxLevelRange][idxIntens]) and (
                idxLevelRange < constantes['N_RAP_RANGES'] - 1):
            idxLevelRange += 1
        CorrLevel = ThirdOctaveLevel[idxIntens] + DLL[idxLevelRange][idxIntens]
        ThirdOctaveIntens[idxIntens] = np.power(10, CorrLevel / 10)
    return ThirdOctaveIntens


# Niveles de las primeras bandas criticas
def f_calc_lcbs(ThirdOctaveIntens):
    # Aproximación a bandas críticas por debajo de 300 Hz.
    LCB = np.zeros(3)
    pcbi = np.zeros(3)
    for idxIntens in range(6):
        pcbi[0] += ThirdOctaveIntens[idxIntens]
    for idxIntens in range(6, 9):
        pcbi[1] += ThirdOctaveIntens[idxIntens]
    for idxIntens in range(9, 11):
        pcbi[2] += ThirdOctaveIntens[idxIntens]
    for i in range(3):
        if pcbi[i] > 0:
            LCB[i] = 10 * np.log10(pcbi[i])
    return LCB


# Cálculo del core loudness
def f_calc_core_loudness(ThirdOctaveSPL, LCB, SoundFieldDiffuse=constantes['SOUNDFIELDDIFFUSE']):
    coreLoudness = np.zeros(constantes['N_CORE_LOUDN'])
    criticalBandLevels = np.zeros(constantes['N_CORE_LOUDN'])
    # Características de transmisión del oído humano
    A0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5, 2.0, 5.0, 12.0])
    # Diferencia de nivel para campo difuso
    DDF = np.array([0, 0, 0.5, 0.9, 1.2, 1.6, 2.3, 2.8, 3.0, 2.0, 0, -1.4, -2.0, -1.9, -1.0, 0.5, 3.0, 4.0, 4.3, 4.0])
    # Umbrales de silencio
    LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    # Corrección por aproximación
    DCB = np.array([-0.25, -0.6, -0.8, -0.8, -0.5, 0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5, 0,
                    -0.5])
    # Factor umbral
    S = 0.25

    for idxCL in range(constantes['N_CORE_LOUDN'] - 1):
        if idxCL < 3:
            Le = LCB[idxCL]
        else:
            Le = ThirdOctaveSPL[idxCL + 8]

        Le = Le - A0[idxCL]
        coreLoudness[idxCL] = 0

        # Corrección por campo difuso
        if SoundFieldDiffuse == 1:
            Le = Le + DDF[idxCL]

        # Correción si se supera el umbral de silencio
        if Le > LTQ[idxCL]:
            Le = Le - DCB[idxCL]
            C1 = 0.0635 * 10 ** (0.025 * LTQ[idxCL])
            C2 = ((1 - S + S * 10 ** (0.1 * (Le - LTQ[idxCL]))) ** 0.25) - 1
            coreLoudness[idxCL] = C1 * C2
            if coreLoudness[idxCL] <= 0:
                coreLoudness[idxCL] = 0
        criticalBandLevels[idxCL] = Le
    return coreLoudness


# Corrección del loudness específico de la banda más baja por silencio en esta banda
def f_corr_loudness(coreLoudness):
    CorrCL = 0.4 + 0.32 * np.float(np.power(coreLoudness[0], 0.2))
    if CorrCL < 1:
        coreLoudness[0] *= CorrCL
    return coreLoudness


# Cálculo del loudness específico con saltos Bark 0.1
def calc_slopes(coreLoudness):
    N1 = 0
    Z = 0.1
    Z1 = 0
    idxRNS = 0
    idxNS = 0
    Loudness = 0
    specLoudness = np.zeros(constantes['N_BARK_BANDS'])

    # Relación bandas tercio de octava y Bark
    ZUP = np.array([0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8,
                    22.7, 23.6, 24.0])
    # Inclinación de la pendiente
    USL = np.array([[13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
                    [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
                    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
                    [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
                    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
                    [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
                    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
                    [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
                    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
                    [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
                    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
                    [0.59, 0.53, 0.51, 0.5, 0.42, 0.42, 0.42, 0.42],
                    [0.4, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
                    [0.27, 0.21, 0.2, 0.18, 0.17, 0.17, 0.17, 0.17],
                    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
                    [0.12, 0.11, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08],
                    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
                    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])
    # Rango del valor de la sonoridad
    RNS = np.array([21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.3, 0.22, 0.15, 0.1, 0.035,
                    0.0])

    # Cálculo de sonoridad específica y total
    for idxCL in range(constantes['N_CORE_LOUDN']):
        idxCBN = idxCL - 1
        ZUP[idxCL] += 0.0001
        if idxCBN > constantes['N_CB_RANGES'] - 1:
            idxCBN = constantes['N_CB_RANGES'] - 1
        nextCriticalBand = 0
        while not nextCriticalBand:
            if N1 > coreLoudness[idxCL]:
                N2 = RNS[idxRNS]
                if N2 < coreLoudness[idxCL]:
                    N2 = coreLoudness[idxCL]
                DZ = (N1 - N2) / USL[idxRNS][idxCBN]
                Z2 = Z1 + DZ
                if Z2 > ZUP[idxCL]:
                    nextCriticalBand = 1
                    Z2 = ZUP[idxCL]
                    DZ = Z2 - Z1
                    N2 = N1 - DZ * USL[idxRNS][idxCBN]
                Loudness += DZ * (N1 + N2) / 2
                ZK = Z
                while ZK <= Z2:
                    specLoudness[idxNS] = N1 - (ZK - Z1) * USL[idxRNS][idxCBN]
                    idxNS += 1
                    ZK = ZK + 0.1
                Z = ZK

            else:
                if N1 < coreLoudness[idxCL]:
                    idxRNS = 0
                    while (idxRNS < constantes['N_RNS_RANGES']) and (RNS[idxRNS] >= coreLoudness[idxCL]):
                        idxRNS += 1
                nextCriticalBand = 1
                Z2 = ZUP[idxCL]
                N2 = coreLoudness[idxCL]
                Loudness += N2 * (Z2 - Z1)
                ZK = Z
                while ZK <= Z2:
                    specLoudness[idxNS] = N2
                    idxNS += 1
                    ZK = ZK + 0.1
                Z = ZK

            while (N2 <= RNS[idxRNS]) and (idxRNS < constantes['N_RNS_RANGES'] - 1):
                idxRNS += 1
            if idxRNS > constantes['N_RNS_RANGES'] - 1:
                idxRNS = constantes['N_RNS_RANGES'] - 1
            Z1 = Z2
            N1 = N2

        if Loudness < 0:
            Loudness = 0
    return Loudness, specLoudness


# Cálculo del loudness específico con saltos Bark 1
def calc_slopes_Bark(coreLoudness):
    N1 = 0
    Z = 1
    Z1 = 0
    idxRNS = 0
    idxNS = 0
    Loudness = 0
    specLoudness = np.zeros(variablesBark['N_BARK_BANDS'])

    # Relación bandas tercio de octava y Bark
    ZUP = np.array([0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6, 21.8,
                    22.7, 23.6, 24.0])
    # Inclinación de la pendiente
    USL = np.array([[13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
                    [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
                    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
                    [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
                    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
                    [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
                    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
                    [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
                    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
                    [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
                    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
                    [0.59, 0.53, 0.51, 0.5, 0.42, 0.42, 0.42, 0.42],
                    [0.4, 0.33, 0.26, 0.24, 0.22, 0.22, 0.22, 0.22],
                    [0.27, 0.21, 0.2, 0.18, 0.17, 0.17, 0.17, 0.17],
                    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
                    [0.12, 0.11, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08],
                    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
                    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02]])
    # Rango del valor de la sonoridad
    RNS = np.array([21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.3, 0.22, 0.15, 0.1, 0.035,
                    0.0])

    # Cálculo de sonoridad específica y total
    for idxCL in range(constantes['N_CORE_LOUDN']):
        idxCBN = idxCL - 1
        ZUP[idxCL] += 0.0001
        if idxCBN > constantes['N_CB_RANGES'] - 1:
            idxCBN = constantes['N_CB_RANGES'] - 1
        nextCriticalBand = 0
        while not nextCriticalBand:
            if N1 > coreLoudness[idxCL]:
                N2 = RNS[idxRNS]
                if N2 < coreLoudness[idxCL]:
                    N2 = coreLoudness[idxCL]
                DZ = (N1 - N2) / USL[idxRNS][idxCBN]
                Z2 = Z1 + DZ
                if Z2 > ZUP[idxCL]:
                    nextCriticalBand = 1
                    Z2 = ZUP[idxCL]
                    DZ = Z2 - Z1
                    N2 = N1 - DZ * USL[idxRNS][idxCBN]
                Loudness += DZ * (N1 + N2) / 2
                ZK = Z
                while ZK <= Z2:
                    specLoudness[idxNS] = N1 - (ZK - Z1) * USL[idxRNS][idxCBN]
                    idxNS += 1
                    ZK = ZK + 1
                Z = ZK

            else:
                if N1 < coreLoudness[idxCL]:
                    idxRNS = 0
                    while (idxRNS < constantes['N_RNS_RANGES']) and (RNS[idxRNS] >= coreLoudness[idxCL]):
                        idxRNS += 1
                nextCriticalBand = 1
                Z2 = ZUP[idxCL]
                N2 = coreLoudness[idxCL]
                Loudness += N2 * (Z2 - Z1)
                ZK = Z
                while ZK <= Z2:
                    specLoudness[idxNS] = N2
                    idxNS += 1
                    ZK = ZK + 1
                Z = ZK

            while (N2 <= RNS[idxRNS]) and (idxRNS < constantes['N_RNS_RANGES'] - 1):
                idxRNS += 1
            if idxRNS > constantes['N_RNS_RANGES'] - 1:
                idxRNS = constantes['N_RNS_RANGES'] - 1
            Z1 = Z2
            N1 = N2

        if Loudness < 0:
            Loudness = 0
    return Loudness, specLoudness


# Conversión sonos a fonos
def sone2phon(Loudness):
    if Loudness >= 1:
        LN = 40 + 33.22 * np.log10(Loudness)
    else:
        LN = 40 * np.power(Loudness + 0.0005, 0.35)
    return LN

# Decaimiento temporal no lineal
def f_nl(coreLoudness):
    # Inicializacion variables
    Tvar = constantes['TVAR']
    Tshort = constantes['TSHORT']
    Tlong = constantes['TLONG']
    deltaT = 1 / (constantes['SR_LEVEL'] * constantes['NL_ITER'])
    P = (Tvar + Tlong) / (Tvar * Tshort)
    Q = 1 / (Tshort * Tvar)
    Lambda1 = -P / 2 + np.sqrt(P * P / 4 - Q)
    Lambda2 = -P / 2 - np.sqrt(P * P / 4 - Q)
    Den = Tvar * (Lambda1 - Lambda2)
    E1 = np.exp(Lambda1 * deltaT)
    E2 = np.exp(Lambda2 * deltaT)
    B = np.array([(E1 - E2) / Den,
                  ((Tvar * Lambda2 + 1) * E1 - (Tvar * Lambda1 + 1) * E2) / Den,
                  ((Tvar * Lambda1 + 1) * E1 - (Tvar * Lambda2 + 1) * E2) / Den,
                  (Tvar * Lambda1 + 1) * (Tvar * Lambda2 + 1) * (E1 - E2) / Den,
                  np.exp(-deltaT / Tlong),
                  np.exp(-deltaT / Tvar)])

    NlaLpDta = {
        'B': B,
        'UoLast': 0,
        'U2Last': 0
    }

    for idxCL in range(constantes['N_CORE_LOUDN']):
        NlaLpDta['UoLast'] = 0
        NlaLpDta['U2Last'] = 0

        for idxTime in range(np.shape(coreLoudness)[1] - 1):
            Delta = (coreLoudness[idxCL, idxTime + 1] - coreLoudness[idxCL, idxTime]) / float(constantes['NL_ITER'])
            Ui = coreLoudness[idxCL, idxTime]
            NlaLpDta = f_nl_lp(Ui, NlaLpDta)
            coreLoudness[idxCL, idxTime] = NlaLpDta['UoLast']
            Ui += Delta
            for idxI in range(constantes['NL_ITER'] - 1):
                NlaLpDta = f_nl_lp(Ui, NlaLpDta)
                Ui += Delta
        NlaLpDta = f_nl_lp(coreLoudness[idxCL, idxTime + 1], NlaLpDta) # Actualizar diccionario
        coreLoudness[idxCL, idxTime + 1] = NlaLpDta['UoLast']
    return coreLoudness


# Actualización parámetros para cálculo de decaimiento temporal
def f_nl_lp(Ui, NlaLpDta):
    if Ui < NlaLpDta['UoLast']:     # Caso 1
        if NlaLpDta['UoLast'] > NlaLpDta['U2Last']:     # Caso 1.1
            U2 = NlaLpDta['UoLast'] * NlaLpDta['B'][0] - NlaLpDta['U2Last'] * NlaLpDta['B'][1]
            Uo = NlaLpDta['UoLast'] * NlaLpDta['B'][2] - NlaLpDta['U2Last'] * NlaLpDta['B'][3]
            if Uo < Ui:     # Uo no puede ser menor que Ui
                Uo = Ui
            if U2 > Uo:     # U2 no puede ser mayor que Uo
                U2 = Uo
        else:       # Caso 1.2
            Uo = NlaLpDta['UoLast'] * NlaLpDta['B'][4]
            if Uo < Ui:     # Uo no puede ser menor que Ui
                Uo = Ui
            U2 = Uo
    else:
        if Ui == NlaLpDta['UoLast']:        # Caso 2
            Uo = Ui
            if Uo > NlaLpDta['U2Last']:     # Caso 2.1
                U2 = (NlaLpDta['U2Last'] - Ui) * NlaLpDta['B'][5] + Ui
            else:       # Caso 2.2
                U2 = Ui
        else:       # Caso 3
            Uo = Ui
            U2 = (NlaLpDta['U2Last'] - Ui) * NlaLpDta['B'][5] + Ui
    NlaLpDta['UoLast'] = Uo
    NlaLpDta['U2Last'] = U2
    return NlaLpDta


# Ponderación temporal
def loudness_zwicker_temporal_weighting(loudness):
    RATE = 2000
    Tau = 3.5 * 10**-3
    Filt1 = loudness_zwicker_lowpass_intp(loudness, 
                                          Tau, 
                                          RATE)
    Tau = 70 * 10**-3
    Filt2 = loudness_zwicker_lowpass_intp(loudness, 
                                          Tau, 
                                          RATE)
    Loudness = 0.47 * Filt1 + 0.53 * Filt2
    return Loudness


# Filtro paso bajo e interpolación para obtenr frecuancia de muetreo de 48 kHz
def loudness_zwicker_lowpass_intp(loudness, Tau, RATE=2000):
    numSamples = np.shape(loudness)[0]
    Y = 0
    Input = loudness.copy()
    Output = np.zeros(numSamples)

    A1 = np.exp(-1 / (RATE * constantes['LP_ITER'] * Tau))
    B0 = 1 - A1

    for idxTime in range(numSamples):
        Output[idxTime] = B0 * Input[idxTime] + A1 * Y
        Y = Output[idxTime]
        if idxTime < numSamples - 1:
            Xd = (Input[idxTime + 1] - Input[idxTime]) / constantes['LP_ITER']
            # Iteraciones internas / interpolacion
            for idxI in range(constantes['LP_ITER']):
                Input[idxTime] += Xd
                Y = B0 * Input[idxTime] + A1 * Y
    return Output


# Función principal sonidos estacionarios
def loudness_ISO532(ThirdOctaveLevels, SoundFieldDiffuse):
    ThirdOctaveIntens = f_corr_third_octave_intensities(ThirdOctaveLevels)
    LCB = f_calc_lcbs(ThirdOctaveIntens)
    coreLoudness = f_calc_core_loudness(ThirdOctaveLevels, 
                                        LCB, 
                                        SoundFieldDiffuse)
    coreLoudness = f_corr_loudness(coreLoudness)
    Loudness, specLoudness = calc_slopes(coreLoudness)
    LoudnessBark, specLoudnessBark = calc_slopes_Bark(coreLoudness)
    return Loudness, specLoudness, LoudnessBark, specLoudnessBark


# Función principal sonidos variantes en el tiempo
def loudness_ISO532_time(ThirdOctaveLevels, SoundFieldDiffuse, RATE=48000, CHUNK=4800):
    numSampleLevel = np.shape(ThirdOctaveLevels)[1]
    coreLoudness = np.zeros((21, numSampleLevel))
    for idxTime in range(numSampleLevel):
        ThirdOctaveIntens = f_corr_third_octave_intensities(ThirdOctaveLevels[:, idxTime])
        LCB = f_calc_lcbs(ThirdOctaveIntens)
        coreLoudness[:, idxTime] = f_calc_core_loudness(ThirdOctaveLevels[:, idxTime], 
                                                        LCB, 
                                                        SoundFieldDiffuse)
        coreLoudness[:, idxTime] = f_corr_loudness(coreLoudness[:, idxTime])
    coreLoudness = f_nl(coreLoudness)
    Loudness = np.zeros(numSampleLevel)
    specLoudness = np.zeros((constantes['N_BARK_BANDS'], 
                            numSampleLevel))
    for idxTime in range(numSampleLevel):
        Loudness[idxTime], specLoudness[:, idxTime] = calc_slopes(coreLoudness[:, idxTime])
    filtLoudness = loudness_zwicker_temporal_weighting(Loudness)
    return filtLoudness[::constantes['DEC_FACTOR']], specLoudness[:, ::constantes['DEC_FACTOR']]