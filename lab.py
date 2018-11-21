#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# numero de 
# taps (coeficientes do filtro)

# Filtro passa-baixas - Obtém apenas o som do carro.
def car_filter():
    numtaps = 400
    fc = 1500
    [fs,x] = wavfile.read("fontes.wav")
    print(fs)

    Y = np.fft.fft(x, 22050)
    plt.plot(Y)
    plt.show()

    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequencia Hz')
    plt.xlabel('tempo s')
    plt.show()

    # calcula um filtro FIR passa-baixas com janela de hamming
    A = 1
    B = signal.firwin(numtaps, float(fc) / float(fs) * 2, window='hamming')
    plt.plot(B)
    plt.show()

    # calcula a resposta em frequencia do filtro
    FFTB = np.fft.fft(B, fs)
    plt.plot(abs(FFTB))
    plt.show()

    y = signal.lfilter(B,A,x)
    #y = np.convolve(x,B)
    plt.plot(y)
    plt.show()

    wavfile.write('carro.wav', fs, np.int16(y))

# Filtro passa-faixas
def whistle_filter():
    numtaps = 500
    f1 = 5000
    f2 = 5650 
    [fs,x] = wavfile.read("fontes.wav")
    print(fs)

    Y = np.fft.fft(x, 22050)
    plt.plot(Y)
    plt.show()

    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequencia Hz')
    plt.xlabel('tempo s')
    plt.show()

    # Cálculo para um filtro FIR Passa-faixas
    A = 1
    B = signal.firwin(numtaps, [0.47, 0.52], pass_zero=False)
    plt.plot(B)
    plt.show()

    # calcula a resposta em frequencia do filtro
    FFTB = np.fft.fft(B, fs)
    plt.plot(abs(FFTB))
    plt.show()

    y = signal.lfilter(B,A,x)
    #y = np.convolve(x,B)
    plt.plot(y)
    plt.show()

    wavfile.write('apito.wav', fs, np.int16(y))

# Filtro Multi-band
def bird_filter():
    numtaps = 800
    f1 = 5000
    f2 = 5650 
    [fs,x] = wavfile.read("fontes.wav")
    print(fs)

    Y = np.fft.fft(x, 22050)
    plt.plot(Y)
    plt.show()

    f, t, Sxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequencia Hz')
    plt.xlabel('tempo s')
    plt.show()

    # Cálculo filtro FIR Multi-band
    A = 1
    B = signal.firwin(numtaps, [0.17, 0.43, 0.6, 0.77], pass_zero=False)
    plt.plot(B)
    plt.show()

    # calcula a resposta em frequencia do filtro
    FFTB = np.fft.fft(B, fs)
    plt.plot(abs(FFTB))
    plt.show()

    y = signal.lfilter(B,A,x)
    #y = np.convolve(x,B)
    plt.plot(y)
    plt.show()

    wavfile.write('passaros.wav', fs, np.int16(y))


# Chamada das funções
#car_filter()
#bird_filter()
#whistle_filter()

# Cálculo da fft
#y = x[9541:14200]
#Y = np.fft.fft(y, 8000)
	
#plt.plot(abs(Y))
#plt.plot(x)
#plt.show()
#os.system("aplay pno1.wav")