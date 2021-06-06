# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Algoritmo para la detección de voz

import numpy as np
import statistics
import pyaudio

# Detección de voz
def VAD(frames):
    speech = None
    Energy_PrimThresh = 40
    F_PrimThresh = 185
    SF_PrimThresh = 5
    silenceCount = 0
    speechCount = 0

    for frame in frames:
        Ei = shortTermEnergy(frame)
        Fi = dominantFrequency(frame)
        SFMi = SFM(frame)

        E_min = np.min(Ei)
        F_min = np.min(Fi)
        SF_min = np.min(SFMi)

        Thresh_E = Energy_PrimThresh*np.log10(E_min)
        Thresh_F = F_PrimThresh
        Thresh_SF = SF_PrimThresh

        count = 0

        if (Ei - E_min) >= Thresh_E:
            count += 1
        if (Fi - F_min) >= Thresh_F:
            count += 1
        if (SFMi - SF_min) >= Thresh_SF:
            count += 1

        if count > 1:
            speechCount += 1
            silenceCount = 0
        else:
            E_min = (silenceCount*E_min + Ei) / (silenceCount + 1)
            silenceCount += 1
            speechCount = 0
        Thresh_E = Energy_PrimThresh*np.log10(E_min)

        if speechCount >= 5:
            speech = True

        return speech


# Cálculo de la energía a corto plazo
def shortTermEnergy(frame):
    return sum([abs(x) ** 2 for x in frame]) / len(frame)


# Cálculo de la frecuencia dominante
def dominantFrequency(frame, CHUNK = 480, RATE = 48000):
    fft_wave = np.fft.fft(frame)
    idx = np.argmax(abs(fft_wave))
    freq = np.fft.fftfreq(CHUNK, 1)
    return abs(RATE*freq[int(idx)])


# Cálculo medida de planitud espectral
def SFM(frame, CHUNK=480):
    fft_wave = np.zeros(CHUNK)
    fft_wave = abs(np.fft.fft(frame))
    am = statistics.mean(fft_wave)
    gm = statistics.geometric_mean(fft_wave)
    return 10*np.log10(gm/am)


# Ejemplo main para detección voz

'''
# Constantes e inicialización pyaudio
CHUNK = 480  # Tamaño en muestras almacenadas en cada array. 10 ms para detección voz.
RATE = 48000  # Muestras por segundo
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, rate=RATE, channels=1, input=True, input_device_index=2,
                frames_per_buffer=CHUNK)  # Input Device = 0 si ordenador, 1 si cascos, 2 si Raspberry


def captarAudio(corr_factor = 4.73):
    # Almacenar audio en array
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
    # Factor de corrección del dispositivo que está siendo utilizado para captar la señal. 94 dB a 1 kHz 1 Pa de referencia.
    data = data * corr_factor
    return data

try:
    # Bucle para captar audio de forma continua
    N = 0
    frames = []
    while True:  
        # Captación audio
        data = captarAudio()
        frames.append(data)
        N += 1

        if N == 50:
           	speech = VAD(frames)
        	if speech:
        		print('Voz detectada')
        	else:
        		print('Silencio')
        	N=0
        	frames=[]

# Salir del bucle con ctr+c
except KeyboardInterrupt:
    # Cierre stream
    stream.stop_stream()
    stream.close()
    p.terminate
    print('Interrupción de teclado. Finalizando programa.')
    sys.exit()
'''