from librosa.util import find_files
from librosa.core import stft, load, istft, resample
from librosa.output import write_wav
import model
import const_nums as C
import numpy as np
import os.path
import torch.nn as nn
from torch.autograd import Variable


# from torch.utils import data
import torch


# Data Processing
def SaveSpectrogram(y_mix, y_vocal, y_inst, fname, original_sr=44100):
    y_mix = resample(y_mix, original_sr, C.SR)
    y_vocal = resample(y_vocal, original_sr, C.SR)
    y_inst = resample(y_inst, original_sr, C.SR)

    S_mix = np.abs(
        stft(y_mix, n_fft=C.WINDOW_SIZE, hop_length=C.HOP_LENGTH)).astype(np.float32)
    S_vocal = np.abs(
        stft(y_vocal, n_fft=C.WINDOW_SIZE, hop_length=C.HOP_LENGTH)).astype(np.float32)
    S_inst = np.abs(
        stft(y_inst, n_fft=C.WINDOW_SIZE, hop_length=C.HOP_LENGTH)).astype(np.float32)

    norm = S_mix.max()
    S_mix /= norm
    S_vocal /= norm
    S_inst /= norm

    # # save the whole spectrogram w/o slicing
    # np.savez(os.path.join(C.PATH_FFT, fname+".npz"),
    #          mix=S_mix, vocal=S_vocal, inst=S_inst)


    # slice signal into PATCH_LEGNTH
    slices_mix = slice_signal(S_mix)
    slices_vocal = slice_signal(S_vocal)
    slices_inst = slice_signal(S_inst)

    for idx, triple in enumerate(zip(slices_mix, slices_vocal, slices_inst)):
        np.savez(os.path.join(C.PATH_FFT, '{}_{}'.format(fname, idx)+".npz"),
            mix=triple[0], vocal=triple[1], inst=triple[2])


# Data Processing
def slice_signal(spectrogram):
    """
    Helper function for slicing the magnitude spectrogram
    by patch length
    # Reference Code는 slice하지 않은 채 저장했다가 TRAINING할 때 전부 불러와서 PATCH_SIZE만큼 넣어줌
    # Reference Code의 장점: 같은 파일 (Time Frame 2000개 중 128개를 랜덤으로 중복해서 학습시킬 수 있음)
    # Reference Code의 단점: 파일을 전부 들고 있어야 됨 (메모리가 큼)
    # My Method: 동우형처럼 아예 저장할 때 부터 PATCH_LENGTH만큼 잘라서 저장함
    # Alternative Methods: 안 자른 상태에서 BATCH_SIZE만큼 파일을 로드하고 Reference처럼 실행 (Reference와 My Method의 중간 단계)
    """
    slices = []
    for end_idx in range(C.PATCH_LENGTH, spectrogram.shape[1], C.PATCH_LENGTH):
        start_idx = end_idx - C.PATCH_LENGTH
        slice_spectrogram = spectrogram[:, start_idx:end_idx]
        slices.append(slice_spectrogram)
    return slices


# for Audio Sepration (get original, vocal, instrument wav files)
def LoadAudio(fname):
    y, _ = load(fname, sr=C.SR)
    spec = stft(y, n_fft=C.WINDOW_SIZE, hop_length=C.HOP_LENGTH)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase


# for Audio Sepration (get original, vocal, instrument wav files)
def SaveAudio(fname, mag, phase):
    y = istft(mag*phase, hop_length=C.HOP_LENGTH, win_length=C.WINDOW_SIZE)
    write_wav(fname, y, C.SR, norm=True)

use_devices = [0, 1, 2, 3]

# for Audio Sepration (get original, vocal, instrument wav files)
def ComputeMask(input_mag, unet_model="unet.model", hard=False):
    unet = nn.DataParallel(model.UNET(), device_ids=use_devices).cuda()

    unet.load_state_dict(torch.load(unet_model))
    # unet.load(unet_model)
    # unet = torch.load(unet_model)
    # print(input_mag.shape)
    # print(input_mag)
    test_var = Variable(torch.from_numpy(input_mag[np.newaxis, np.newaxis, :512, :])).cuda()
    mask = unet(test_var).data[0, 0, :, :]
    mask = np.vstack((np.zeros(mask.shape[1], dtype="float32"), mask))
    if hard:
        hard_mask = np.zeros(mask.shape, dtype="float32")
        hard_mask[mask > 0.5] = 1
        return hard_mask
    else:
        return mask