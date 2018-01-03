from librosa.util import find_files
from librosa.core import stft, load, istft, resample
from librosa.output import write_wav
import network
import const_nums as C
import numpy as np
import os.path
from torch.utils import data
import torch

def LoadDataset(target="vocal"):
    filelist_fft = find_files(C.PATH_FFT, ext="npz")[:200]
    Xlist = []
    Ylist = []
    for file_fft in filelist_fft:
        dat = np.load(file_fft)
        Xlist.append(dat["mix"])
        if target == "vocal":
            assert(dat["mix"].shape == dat["vocal"].shape)
            Ylist.append(dat["vocal"])
        else:
            assert(dat["mix"].shape == dat["inst"].shape)
            Ylist.append(dat["inst"])
    return Xlist, Ylist