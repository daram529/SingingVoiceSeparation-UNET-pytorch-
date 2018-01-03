from librosa.util import find_files
import torch
from torch.utils import data
import numpy as np
import os
import const_nums as C
from torch.autograd import Variable

class LoadDataset(data.Dataset):
    """
    Audio sample reader.
    Used alongside with DataLoader class to generate batches.
    see: http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset
    """
    def __init__(self, data_folder_path=C.PATH_FFT):
        if not os.path.exists(data_folder_path):
            raise Error('The data folder does not exist!')

        # store full paths - not the actual files.
        # all files cannot be loaded up to memory due to its large size.
        # insted, we read from files upon fetching batches (see __getitem__() implementation)
        self.filepath_list_fft = find_files(data_folder_path, ext="npz")
        self.num_data = len(self.filepath_list_fft)

    def test_data(self, num_test_audio):
        """
        Randomly chosen batch for testing generated results.
        Args:
            num_test_audio(int): number of test audio.
                Must be same as batch size of training,
                otherwise it cannot go through the forward step of generator.
        """
        test_filenames = np.random.choice(self.filepath_list_fft, num_test_audio)
        test_mixtures = [np.load(f)["mix"] for f in test_filenames]
        test_basenames = [os.path.basename(fpath) for fpath in test_filenames]
        test_phases = [np.load(f)["phase"] for f in test_filenames]
        return test_basenames, test_mixtures, test_phases

    # for data load (training model)
    def split_pair_to_vars(batch_pair, target="vocal"):
        # mixed_batch = []
        # target_batch = []
        # for pair in batch_pair:
        #     mixed_batch.append(pair[0])
        #     if target == "vocal":
        #         assert(pair[0].shape == pair[1].shape)
        #         target_batch.append(pair[1])
        #     else:
        #         assert(pair[0].shape == pair[2].shape)
        #         target_batch.append(pair[2])

        # print(len(mixed_batch))

        print(batch_pair)
        print(len(batch_pair))
        mixed_batch = batch_pair[0]
        target_batch = batch_pair[1]
        print(len(mixed_batch))

        mixed_batch_var = Variable(torch.from_numpy(np.array(mixed_batch)[:, :512, :].reshape(C.BATCH_SIZE, 512, 128, 1))).cuda()
        target_batch_var = Variable(torch.from_numpy(np.array(target_batch)[:, :512, :].reshape(C.BATCH_SIZE, 512, 128, 1))).cuda()

        return mixed_batch_var, target_batch_var

    def __getitem__(self, idx):
        # get item for specified index
        pair = np.load(self.filepath_list_fft[idx])
        pair_list = [pair["mix"], pair["vocal"], pair["inst"]]
        return pair_list

    def __len__(self):
        return self.num_data