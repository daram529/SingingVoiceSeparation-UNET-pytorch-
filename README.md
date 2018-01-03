# SingingVoiceSeparation-UNET-pytorch

# Chainer: https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer

*If you want to train U-Net with your own dataset, prepare the mixed, instrumental-only, and vocal-only versions of each track, and pickle their spectrograms using util.SaveSpectrogram() function. You should set PATH_FFT (in const.py) to the directory you want to save the pickled data.

*If you have either iKala, MedleyDB, DSD100 dataset, you could make use of ProcessXX.py scripts. Remember to set the PATH_XX in each script to the right path.

*If you want to generate dataset with "original" and "instrumental version" audio pairs (as the original work did), refer to ProcessIMAS.py.