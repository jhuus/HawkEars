# Model training

import argparse
import logging
import time

from core import cfg, configs, data_module, set_config
from model import main_model

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self):
        torch.set_float32_matmul_precision('medium')
        if not cfg.train.seed is None:
            pl.seed_everything(cfg.train.seed, workers=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.train.seed)

        if cfg.train.deterministic:
            cfg.train.num_workers = 0
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run(self):
        # load all the data once for performance, then split as needed in each fold
        dm = data_module.DataModule()
        dm.load_data()
        weights = dm.class_weights()

        for k in range(cfg.train.num_folds):
            trainer = pl.Trainer(
                devices=1,
                accelerator='auto',
                callbacks=[ModelCheckpoint(save_top_k=cfg.train.save_last_n, mode='max', monitor='epoch_num'),
                           TQDMProgressBar(refresh_rate=10)],
                deterministic=cfg.train.deterministic,
                max_epochs=cfg.train.num_epochs,
                precision='16-mixed' if cfg.train.mixed_precision else 32,
                logger=TensorBoardLogger(save_dir='logs', name=f'fold-{k}', default_hp_metric=False),
            )

            dm.prepare_fold(k, cfg.train.num_folds)

            # create model inside loop so parameters are reset for each fold,
            # and so metrics are tracked correctly
            model = main_model.MainModel(dm.train_class_names, dm.train_class_codes, dm.test_class_names, weights, cfg.train.model_name, cfg.train.load_weights, num_train_specs=dm.num_train_specs)

            if cfg.train.compile:
                model = torch.compile(model)

            trainer.fit(model, dm)

            if cfg.misc.test_pickle is not None:
                trainer.test(model, dm)

if __name__ == '__main__':
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='base', help=f"Configuration name. Default = 'base'.")
    parser.add_argument('-de', '--debug', default=False, action='store_true', help="Flag for debug mode.")
    parser.add_argument('-e', '--epochs', type=int, default=None, help=f"Number of epochs.")
    args = parser.parse_args()

    cfg_name = args.config
    if cfg_name in configs:
        set_config(cfg_name)
    else:
        print(f"Configuration '{cfg_name}' not found.")
        quit()

    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")

    start_time = time.time()
    Trainer().run()
    elapsed = time.time() - start_time
    logging.info(f"Elapsed time = {int(elapsed) // 60}m {int(elapsed) % 60}s\n")
