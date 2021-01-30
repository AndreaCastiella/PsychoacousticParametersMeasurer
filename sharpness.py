import numpy as np


def calc_sharpness(loudness, specificLoudness):
    n = len(specificLoudness)

    # Zwicker method
    gz_Z = np.ones(n)
    for z in range(160, n):
        gz_Z[z] = 0.066 * np.e ** (0.171 * (z / 10))
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += specificLoudness[i] * gz_Z[i] * round(z, 1) * 0.1
    sharpnessZwicker = 0.11 * suma / loudness

    # Von Bismarck method
    gz_VB = np.ones(n)
    for z in range(140, n):
        gz_VB[z] = 0.00012 * (z / 10) ** 4 - 0.0056 * (z / 10) ** 3 + 0.1 * (z / 10) ** 2 - 0.81 * (z / 10) + 3.5
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += specificLoudness[i] * gz_VB[i] * round(z, 1) * 0.1
    sharpnessVonBismarck = 0.11 * suma / loudness

    # Aures method
    gz_A = np.ones(n)
    for z in range(160, n):
        gz_A[z] = 0.066 * np.e ** (0.171 * (z / 10))
    suma = 0
    for i, z in enumerate(np.arange(0.1, n / 10 + 0.1, 0.1)):
        suma += (specificLoudness[i] * gz_A[i] * 0.1)
    sharpnessAures = 0.11 * (suma / np.log(0.05 * loudness) + 1)

    return round(sharpnessZwicker, 4), round(sharpnessVonBismarck, 4), round(sharpnessAures, 4)
