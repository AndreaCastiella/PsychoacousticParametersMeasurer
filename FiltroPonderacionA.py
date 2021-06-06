# -*- coding: utf-8 -*-
# Andrea Castiella Aguirrezabala
# Diseño filtro ponderado A
# Definición de filtro ponderado A segun IEC/CD 1672

from numpy import pi, convolve
from scipy import signal
from scipy.signal import lfilter

# Obtener coeficientes filtro ponderación A
def coeficientesPonderacionA(RATE=48000):
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2*pi*f4)**2*(10**(A1000/20)), 0, 0, 0, 0]
    DENs = convolve([1., +4*pi * f4, (2*pi * f4)**2],
        [1., +4*pi * f1, (2*pi * f1)**2])
    DENs = convolve(convolve(DENs, [1., 2*pi * f3]),
        [1., 2*pi * f2])

    # Uso de una transformación bilineal para obtener el filtro digital
    return signal.bilinear(NUMs, DENs, RATE)


# Filtrar señal para ponderación A.
def filtroPonderacionA(data, RATE=48000):
    b, a = coeficientesPonderacionA(RATE)
    pond_A = lfilter(b, a, data)
    return pond_A