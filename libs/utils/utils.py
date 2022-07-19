import os
import time
import shutil
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np



def set_gpu(gpu):
    print('set gpu: {:s}'.format(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: {:s}'.format(path))


def check_path(path):
    if not os.path.exists(path):
        raise ValueError('path does not exist: {:s}'.format(path))


def ensure_path(path, remove=False):
    if os.path.exists(path):
        if remove:
            if input('{:s} exists, remove? ([y]/n): '.format(path)) != 'n':
                shutil.rmtree(path)
                os.makedirs(path)
    else:
        os.makedirs(path)


def fix_random_seed(seed, reproduce=False):
    cudnn.enabled = True
    cudnn.benchmark = True

    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng
