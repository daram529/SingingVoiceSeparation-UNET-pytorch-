import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os
import numpy as np
from model_parts import *
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from load_dataset import LoadDataset
import const_nums as C
import util


class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()                    # input: [B x 1 x 512 x 128]
        # # encoder
        self.down1 = down(1, 16)                        # output: [B x 16 x 256 x 64]
        self.down2 = down(16, 32)                       # output: [B x 32 x 128 x 32]
        self.down3 = down(32, 64)                       # output: [B x 64 x 64 x 16]
        self.down4 = down(64, 128)                      # output: [B x 128 x 32 x 8]
        self.down5 = down(128, 256)                     # output: [B x 256 x 16 x 4]
        self.down6 = down(256, 512)                     # output: [B x 512 x 8 x 2]

        #decoder
        self.up1 = up(512, 256, dropout_flag=True)      # output: [B x 256 x 16 x 4]
        self.up2 = up(512, 128, dropout_flag=True)      # output: [B x 128 x 32 x 8]
        self.up3 = up(256, 64, dropout_flag=True)       # output: [B x 64 x 64 x 16]
        self.up4 = up(128, 32)                          # output: [B x 32 x 128 x 32]
        self.up5 = up(64, 16)                           # output: [B x 16 x 256 x 64]
        self.fconv = final_conv(32, 1)                  # output: [B x 1 x 512 x 128]
        # self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, X):
        # x1 = self.relu1(self.batch1(self.conv1(X)))
        x1 = self.down1(X)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x = self.up1(x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.fconv(x, x1)
        return x

    def save(self, fname):
        torch.save(self.state_dict(), fname)

    def load(self, fname):
        self.load_state_dict(torch.load(fname))


def split_pair_to_vars(batch_pairs):
    mixed_batch = batch_pairs[0]
    target_batch = batch_pairs[1]

    # print(mixed_batch[:, :512, :])
    # print(mixed_batch[:, :512, :].reshape(C.BATCH_SIZE, 512, 128, 1))

    # print(mixed_test.shape[0])
    # print(mixed_batch.numpy().shape)


    mixed_batch_process = mixed_batch[:, :512, :].numpy().reshape(C.BATCH_SIZE, 1, 512, 128)
    target_batch_process = target_batch[:, :512, :].numpy().reshape(C.BATCH_SIZE, 1, 512, 128)

    mixed_batch_var = Variable(torch.from_numpy(mixed_batch_process)).cuda()
    target_batch_var = Variable(torch.from_numpy(target_batch_process)).cuda()

    return mixed_batch_var, target_batch_var

    # Variable(torch.from_numpy(np.array(batch_pairs[0])[:, :512, :].reshape(C.BATCH_SIZE, 512, 128, 1))).cuda()

use_devices = [0, 1, 2, 3]

def TrainModel(epochs=100, load_model=None):
    unet = torch.nn.DataParallel(UNET(), device_ids=use_devices).cuda()  # use GPU
    if load_model:
        unet.load(load_model)

    optimizer = optim.Adam(unet.parameters())
    print('unet created')

    # This is how you define a data loader
    data_loader = LoadDataset(C.PATH_FFT)
    random_data_loader = DataLoader(
            dataset=data_loader,
            batch_size=C.BATCH_SIZE,  # specified batch size here
            shuffle=True,
            num_workers=4,
            drop_last=True,  # drop the last batch that cannot be divided by batch_size
            pin_memory=True)
    print('DataLoader created')

    # test samples for generation
    test_mixture_filenames, test_mixtures, test_phases = data_loader.test_data(random_data_loader.batch_size)
    test_mixtures_np = np.array(test_mixtures)[:, :512, :].reshape(C.BATCH_SIZE, 1, 512, 128)
    test_mixtures_var = Variable(torch.from_numpy(test_mixtures_np)).cuda()
    print('Test samples loaded')

    print('Starting Training...')
    # start_time = time.time()
    for epoch in range(epochs):
        for i, batch_pairs in enumerate(random_data_loader):

            # batch_pairs: [list of mixed, vocal, instrument]
            # 3 * 64 * 512 * 128

            # using the batch pair, split into
            # batch of Mixed Spectrogram and Target Spectrogram
            mixed_batch_var, target_batch_var = split_pair_to_vars(batch_pairs)
            # print(mixed_batch_var.size())

            
            optimizer.zero_grad()
            outputs = unet(mixed_batch_var)
            loss = F.l1_loss(outputs*mixed_batch_var, target_batch_var)
            # loss = torch.sum(torch.abs(outputs * mixed_batch_var - target_batch_var))
            loss.backward()
            optimizer.step()


            # print message per 10 steps
            if (i + 1) % 10 == 0:
                print('Epoch {}, Step {}, loss {}'.format(epoch + 1, i + 1, loss.data[0]))

            # save sampled audio at the beginning of each epoch
            if i == 0:
                generated_soft_mask = unet(test_mixtures_var).data[0, 0, :, :]
                print(generated_soft_mask.shape[1])
                mask = np.vstack((np.zeros(generated_soft_mask.shape[1], dtype="float32"), generated_soft_mask))
                util.SaveAudio("audio/vocal-%s-%s" % (test_mixture_filenames, epoch), test_mixtures[0]*mask, test_phases[0])
                unet.save('model/unet_model-{}.pkl'.format(epoch + 1))
    print('Finished Training!')
    torch.save(unet.state_dict(), 'unet_model-fin.pkl')
