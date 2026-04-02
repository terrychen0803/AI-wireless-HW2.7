#!/usr/bin/python
from __future__ import division
import math
import os
import sys
from pathlib import Path

import numpy as np
import numpy.linalg as la
import scipy.interpolate

sqrt = np.sqrt
pi = math.pi

K = 64
CP = K // 4
P = 64  # number of pilot carriers per OFDM block
allCarriers = np.arange(K)
pilotCarriers = allCarriers[::K // P]
dataCarriers = np.delete(allCarriers, pilotCarriers) if P < K else allCarriers
mu = 2
payloadBits_per_OFDM = K * mu
CR = 1
SNRdb = 25
Clipping_Flag = False
CP_flag = False
NoCP = True

_QPSK_mapping_table = {
    (0, 1): (-1 + 1j,), (1, 1): (1 + 1j,),
    (0, 0): (-1 - 1j,), (1, 0): (1 - 1j,),
}
_QPSK_demapping_table = {v: k for k, v in _QPSK_mapping_table.items()}
_QPSK_Constellation = np.array([[-1 + 1j], [1 + 1j], [-1 - 1j], [1 - 1j]])

_16QAM_mapping_table = {
    (0, 0, 1, 0): (-3 + 3j,), (0, 1, 1, 0): (-1 + 3j,), (1, 1, 1, 0): (1 + 3j,), (1, 0, 1, 0): (3 + 3j,),
    (0, 0, 1, 1): (-3 + 1j,), (0, 1, 1, 1): (-1 + 1j,), (1, 1, 1, 1): (1 + 1j,), (1, 0, 1, 1): (3 + 1j,),
    (0, 0, 0, 1): (-3 - 1j,), (0, 1, 0, 1): (-1 - 1j,), (1, 1, 0, 1): (1 - 1j,), (1, 0, 0, 1): (3 - 1j,),
    (0, 0, 0, 0): (-3 - 3j,), (0, 1, 0, 0): (-1 - 3j,), (1, 1, 0, 0): (1 - 3j,), (1, 0, 0, 0): (3 - 3j,),
}
_16QAM_demapping_table = {v: k for k, v in _16QAM_mapping_table.items()}
_16QAM_Constellation = np.array([
    [-3 + 3j], [-1 + 3j], [1 + 3j], [3 + 3j],
    [-3 + 1j], [-1 + 1j], [1 + 1j], [3 + 1j],
    [-3 - 1j], [-1 - 1j], [1 - 1j], [3 - 1j],
    [-3 - 3j], [-1 - 3j], [1 - 3j], [3 - 3j],
])

_64QAM_mapping_table = {}
levels = [-7, -5, -3, -1, 1, 3, 5, 7]
gray3 = {
    (0, 0, 0): -7, (0, 0, 1): -5, (0, 1, 1): -3, (0, 1, 0): -1,
    (1, 1, 0): 1, (1, 1, 1): 3, (1, 0, 1): 5, (1, 0, 0): 7,
}
for br, xr in gray3.items():
    for bi, yi in gray3.items():
        _64QAM_mapping_table[br + bi] = (xr + 1j * yi,)
_64QAM_demapping_table = {v: k for k, v in _64QAM_mapping_table.items()}
_64QAM_Constellation = np.array([[x + 1j * y] for y in levels[::-1] for x in levels])


def Clipping(x, CL):
    sigma = np.sqrt(np.mean(np.square(np.abs(x))))
    CL = CL * sigma
    x_clipped = np.array(x, copy=True)
    clipped_idx = np.abs(x_clipped) > CL
    x_clipped[clipped_idx] = np.divide((x_clipped[clipped_idx] * CL), np.abs(x_clipped[clipped_idx]))
    return x_clipped


def PAPR(x):
    power = np.abs(x) ** 2
    peakp = np.max(power)
    avgp = np.mean(power)
    return 10 * np.log10(peakp / avgp)


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / 2), 2))
    return (2 * bit_r[:, 0] - 1) + 1j * (2 * bit_r[:, 1] - 1)


def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits) / 4), 4))
    bit_mod = []
    for i in range(int(len(bits) / 4)):
        bit_mod.append(list(_16QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits) / 6), 6))
    bit_mod = []
    for i in range(int(len(bits) / 6)):
        bit_mod.append(list(_64QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4, 1))
        min_distance_index = np.argmin(np.abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred, np.array(_QPSK_demapping_table[tuple(_QPSK_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16, 1))
        min_distance_index = np.argmin(np.abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred, np.array(_16QAM_demapping_table[tuple(_16QAM_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64, 1))
        min_distance_index = np.argmin(np.abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred, np.array(_64QAM_demapping_table[tuple(_64QAM_Constellation[min_distance_index])])))
    return X_pred


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time, CP, CP_flag, mu, K):
    if CP_flag is False:
        bits_noise = np.random.binomial(n=1, p=0.5, size=(K * mu,))
        if mu == 2:
            codeword_noise = Modulation(bits_noise)
        elif mu == 4:
            codeword_noise = Modulation_16(bits_noise)
        else:
            codeword_noise = Modulation_64(bits_noise)
        OFDM_time_noise = np.fft.ifft(codeword_noise)
        cp = OFDM_time_noise[-CP:]
    else:
        cp = OFDM_time[-CP:]
    return np.hstack([cp, OFDM_time])


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(np.abs(convolved ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise, sigma2


def removeCP(signal, CP, K):
    return signal[CP:(CP + K)]


def LS_CE(Y, pilotValue, pilotCarriers, K, P, interpolate_method):
    H_est_at_pilots = Y[pilotCarriers] / pilotValue
    if P < K:
        return interpolate(H_est_at_pilots, pilotCarriers, K, interpolate_method)
    return H_est_at_pilots


def MMSE_CE(Y, pilotValue, pilotCarriers, K, P, h, SNR):
    # [ADDED] Completed the TODO block in the original repo.
    snr = 10 ** (SNR / 10.0)
    H_tilde = Y[pilotCarriers] / pilotValue

    Rhh = np.zeros((P, P), dtype=complex)
    Rhp = np.zeros((K, P), dtype=complex)
    tap_idx = np.arange(len(h))
    tap_pow = np.abs(h) ** 2

    for m in range(P):
        for n in range(P):
            Rhh[m, n] = np.sum(tap_pow * np.exp(-1j * 2 * pi * (pilotCarriers[m] - pilotCarriers[n]) * tap_idx / K))

    for m in range(K):
        for n in range(P):
            Rhp[m, n] = np.sum(tap_pow * np.exp(-1j * 2 * pi * (m - pilotCarriers[n]) * tap_idx / K))

    W_MMSE = Rhp @ la.inv(Rhh + np.eye(P) / snr)
    H_MMSE = W_MMSE @ H_tilde
    return H_MMSE, W_MMSE


interpolate_method = 1


def interpolate(H_est, pilotCarriers, K, method):
    if pilotCarriers[0] > 0:
        slope = (H_est[1] - H_est[0]) / (K // P)
        H_est = np.insert(H_est, 0, H_est[0] - slope * (K // P))
        pilotCarriers = np.insert(pilotCarriers, 0, 0)
    if pilotCarriers[len(pilotCarriers) - 1] < (K - 1):
        slope = (H_est[len(H_est) - 1] - H_est[len(H_est) - 2]) / (K // P)
        H_est = np.append(H_est, H_est[len(H_est) - 1] + slope * (K // P))
        pilotCarriers = np.append(pilotCarriers, (K - 1))
    if method == 0:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est, 'linear')
    else:
        H_interpolated = scipy.interpolate.interp1d(pilotCarriers, H_est, 'cubic')
    index = np.arange(K)
    return H_interpolated(index)


def Normalized_FFT_Matrix(K):
    F = np.zeros((K, K), dtype=complex)
    for i in range(K):
        for j in range(K):
            F[i, j] = np.exp(-1j * 2 * pi * i * j / K)
    return F

F = Normalized_FFT_Matrix(K)
FH = np.conj(F).T / K


# [ADDED] Path-safe dataset loading plus synthetic fallback when the official .npy files are absent.
TOOLS_DIR = Path(__file__).resolve().parent
train_path = TOOLS_DIR / 'channel_train.npy'
test_path = TOOLS_DIR / 'channel_test.npy'


def _synthetic_channels(num_samples, L=16, K=64):
    channels = np.zeros((num_samples, L), dtype=np.complex64)
    pdp = np.exp(-np.arange(L) / 3.0)
    pdp = pdp / np.sum(pdp)
    for i in range(num_samples):
        h = (np.random.randn(L) + 1j * np.random.randn(L)) * np.sqrt(pdp / 2.0)
        channels[i, :] = h.astype(np.complex64)
    return channels


if train_path.exists() and test_path.exists():
    channel_train = np.load(train_path)
    channel_test = np.load(test_path)
else:
    print('[INFO] channel_train.npy / channel_test.npy not found. Using synthetic Rayleigh channels instead.')
    channel_train = _synthetic_channels(20000)
    channel_test = _synthetic_channels(5000)

train_size = channel_train.shape[0]
test_size = channel_test.shape[0]


def get_cyclic_and_cutoff_matrix(h):
    H = np.zeros((K, K), dtype=complex)
    A = np.zeros((K, K), dtype=complex)
    h_ = np.flip(np.append(h, np.zeros((K - CP, 1))))
    for i in range(K):
        H[i] = np.roll(h_, i + 1)
        if i < (CP - 1):
            A[i] = np.hstack([np.zeros(K - CP + i + 1), h_[K - CP:K - i - 1]])
    return H, A


def ofdm_simulate(codeword, channelResponse, SNRdb, mu, CP_flag, K, P, CP,
                  pilotValue, pilotCarriers, dataCarriers, Clipping_Flag, ce_flag=False):
    payloadBits_per_OFDM = mu * len(dataCarriers)

    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        if mu == 2:
            QAM = Modulation(bits)
        elif mu == 4:
            QAM = Modulation_16(bits)
        else:
            QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time, CP, CP_flag, mu, K)
    if Clipping_Flag:
        OFDM_withCP = Clipping(OFDM_withCP, 1)

    OFDM_RX, sigma2 = channel(OFDM_withCP, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX, CP, K)

    if ce_flag:
        return np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP)))

    # second symbol with 64-QAM data
    if mu == 2:
        codeword_qam = Modulation(codeword)
    elif mu == 4:
        codeword_qam = Modulation_16(codeword)
    else:
        codeword_qam = Modulation_64(codeword)

    OFDM_time_codeword = IDFT(codeword_qam)
    OFDM_withCP_codeword = addCP(OFDM_time_codeword, CP, True, mu, K)
    OFDM_RX_codeword, _ = channel(OFDM_withCP_codeword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword, CP, K)

    return np.concatenate((
        np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))),
        np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))
    )), sigma2


ISI = np.zeros(K, dtype=complex)
estimated_ISI = np.zeros((K, 1), dtype=complex)


def ofdm_simulate_cp_free(codeword, H, A, FH, SNR, mu, K, P,
                          pilotValue, pilotCarriers, dataCarriers, CE_flag=False):
    global ISI
    payloadBits_per_OFDM = mu * len(dataCarriers)

    if P < K:
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
        if mu == 2:
            QAM = Modulation(bits)
        elif mu == 4:
            QAM = Modulation_16(bits)
        else:
            QAM = Modulation_64(bits)
        OFDM_data = np.zeros(K, dtype=complex)
        OFDM_data[pilotCarriers] = pilotValue
        OFDM_data[dataCarriers] = QAM
    else:
        OFDM_data = pilotValue

    yp = (H - A) @ FH @ OFDM_data
    signal_power = np.mean(np.abs(yp ** 2))
    sigma2 = signal_power * 10 ** (-SNR / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*yp.shape) + 1j * np.random.randn(*yp.shape))
    yp = yp + noise
    yp = yp + ISI
    ISI = A @ FH @ OFDM_data

    if CE_flag:
        return np.concatenate((np.real(yp), np.imag(yp)))

    if mu == 2:
        codeword_qam = Modulation(codeword)
    elif mu == 4:
        codeword_qam = Modulation_16(codeword)
    else:
        codeword_qam = Modulation_64(codeword)

    yd = (H - A) @ FH @ codeword_qam
    yd = yd + noise
    yd = yd + ISI
    ISI = A @ FH @ codeword_qam

    return np.concatenate((
        np.concatenate((np.real(yp), np.imag(yp))),
        np.concatenate((np.real(yd), np.imag(yd)))
    )), sigma2, codeword_qam


def get_WMMSE(SNR, CP_flag=True):
    index = np.random.choice(np.arange(test_size), size=1)
    h = channel_test[index].reshape((-1,))
    H, A = get_cyclic_and_cutoff_matrix(h)
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
    if NoCP:
        signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
    else:
        signal_output = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag, ce_flag=True)
    yp_complex = signal_output[0:K] + 1j * signal_output[K:2 * K]
    Yp_complex = F @ yp_complex
    _, W_MMSE = MMSE_CE(Yp_complex, pilotValue, pilotCarriers, K, P, h, SNR)
    W_MMSE = np.concatenate((
        np.concatenate((np.real(W_MMSE), -np.imag(W_MMSE)), axis=1),
        np.concatenate((np.imag(W_MMSE), np.real(W_MMSE)), axis=1)
    ))
    return W_MMSE


def sample_gen(bs, SNR=20, training_flag=True, NoCP=False, CP_flag=True):
    if training_flag:
        index = np.random.choice(np.arange(train_size), size=bs)
        h_total = channel_train[index]
    else:
        index = np.random.choice(np.arange(test_size), size=bs)
        h_total = channel_test[index]

    H_samples = []
    H_labels = []
    Yp = []

    for h in h_total:
        H_true = np.fft.fft(h, n=K)
        H, A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))

        if NoCP:
            signal_output = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers, CE_flag=True)
        else:
            signal_output = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag, ce_flag=True)

        yp_complex = signal_output[0:K] + 1j * signal_output[K:2 * K]
        Yp_complex = F @ yp_complex
        H_LS = LS_CE(Yp_complex, pilotValue, pilotCarriers, K, P, interpolate_method)

        H_true = np.concatenate((np.real(H_true), np.imag(H_true)))
        H_LS = np.concatenate((np.real(H_LS), np.imag(H_LS)))
        H_labels.append(H_true)
        H_samples.append(H_LS)
        Yp.append(np.concatenate((np.real(Yp_complex), np.imag(Yp_complex))))

    Xp = np.tile(np.concatenate((np.real(pilotValue), np.imag(pilotValue))), (bs, 1))
    return np.asarray(H_samples), np.asarray(H_labels), np.asarray(Yp), np.asarray(Xp)


def test_ce(sess, input_holder, output, SNR, est_type, NoCP=False, CP_flag=True):
    num_trail = 1000
    L = 16
    downsampler = allCarriers[::K // L]
    MSE_T, MSE_F = 0., 0.

    for i in range(num_trail):
        index = np.random.choice(np.arange(test_size), size=1)
        h = channel_test[index].reshape((-1,))
        Htrue = np.fft.fft(h, n=K)
        H, A = get_cyclic_and_cutoff_matrix(h)
        bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))

        if NoCP:
            signal_output, sigma2, _ = ofdm_simulate_cp_free(bits, H, A, FH, SNR, mu, K, P, pilotValue, pilotCarriers, dataCarriers)
        else:
            signal_output, sigma2 = ofdm_simulate(bits, h, SNR, mu, CP_flag, K, P, CP, pilotValue, pilotCarriers, dataCarriers, Clipping_Flag)

        yp_complex = signal_output[0:K] + 1j * signal_output[K:2 * K]
        Yp_complex = F @ yp_complex

        if est_type == 'ls':
            estimated_H = LS_CE(Yp_complex, pilotValue, pilotCarriers, K, P, interpolate_method)
        elif est_type == 'mmse':
            estimated_H, _ = MMSE_CE(Yp_complex, pilotValue, pilotCarriers, K, P, h, SNR)
        elif est_type == 'ce_net':
            H_LS = LS_CE(Yp_complex, pilotValue, pilotCarriers, K, P, interpolate_method)
            H_LS = np.concatenate((np.real(H_LS), np.imag(H_LS))).reshape(1, 2 * K)
            estimated_H = sess.run(output, feed_dict={input_holder: H_LS}).reshape(-1,)
            estimated_H = estimated_H[:K] + 1j * estimated_H[K:2 * K]
        else:  # DNN for CE
            input1 = np.concatenate((np.real(Yp_complex), np.imag(Yp_complex))).reshape(1, 2 * K)
            input2 = np.concatenate((np.real(pilotValue), np.imag(pilotValue))).reshape(1, 2 * K)
            net_input = np.concatenate((input1, input2), axis=1)
            estimated_H = sess.run(output, feed_dict={input_holder: net_input}).reshape(-1,)
            estimated_H = estimated_H[:K] + 1j * estimated_H[K:2 * K]

        estimated_h = np.fft.ifft(estimated_H[downsampler])
        MSE_F += np.sum(np.abs(estimated_H - Htrue) ** 2) / np.sum(np.abs(Htrue) ** 2)
        MSE_T += np.sum(np.abs(estimated_h - h) ** 2) / np.sum(np.abs(h) ** 2)
        sys.stdout.write('\rMSE_T={mse_t:.6f} MSE_F={mse_f:.6f}'.format(
            mse_t=10 * np.log10(MSE_T / (i + 1)),
            mse_f=10 * np.log10(MSE_F / (i + 1))
        ))
        sys.stdout.flush()

    return MSE_T / num_trail, MSE_F / num_trail


Pilot_file_name = str(TOOLS_DIR / ('Pilot_' + str(P) + '_mu' + str(mu) + '.txt'))
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    bits = np.random.binomial(n=1, p=0.5, size=(P * mu,))
    np.savetxt(Pilot_file_name, bits, delimiter=',')

if mu == 2:
    pilotValue = Modulation(bits)
elif mu == 4:
    pilotValue = Modulation_16(bits)
else:
    pilotValue = Modulation_64(bits)
