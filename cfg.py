import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.bs = 4
Cfg.val_bs = 4
Cfg.width = 128
Cfg.height = 128
Cfg.lr = 0.001
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.eta_min = 1e-5
Cfg.t_max = 10
Cfg.total_itrs = 1e6
Cfg.freq=100
Cfg.val_freq = 2000
Cfg.test_freq = 4000
Cfg.gpu_id = '2'
Cfg.test_dir = 'data/test'
Cfg.train_dir = 'data/train'
Cfg.valid_dir = 'data/valid'
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')
Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
