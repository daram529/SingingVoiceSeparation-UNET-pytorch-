#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:52:48 2017

@author: wuyiming
"""

import util
import argparse
import model

"""
Code example for training U-Net
"""


if __name__ == '__main__':
    # parse arguments
    argparser = argparse.ArgumentParser(description='Singing Voice Separation UNet')
    argparser.add_argument('-t', '--train', type=str, default="True", help='train mode')
    argparser.add_argument('-f', '--file', type=str, default="original_mix.wav", help='original mix file to infer')
    args = argparser.parse_args()

    if args.train == "True":
    	model.TrainModel()
    else:
    	"""
    	Code example for performing vocal separation with U-Net
    	"""
    	mag, phase = util.LoadAudio(args.file)
    	start = 1024
    	end = 1024+128
    	print(mag.shape)
    	mask = util.ComputeMask(mag[:, start:end], unet_model="unet_model-fin.pkl", hard=False)

    	util.SaveAudio("vocal-%s" % args.file, mag[:, start:end]*mask, phase[:, start:end])
    	util.SaveAudio("inst-%s" % args.file, mag[:, start:end]*(1-mask), phase[:, start:end])
    	util.SaveAudio("orig-%s" % args.file, mag[:, start:end], phase[:, start:end])