import os
import pytorch_lightning as pl
from os.path import join


import random
import torch 
import numpy as np


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ImdbDataModule, BertTextClassificationSystem
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json
from metaflow import FlowSpec, step, Parameter

class TrainClassifier(FlowSpec):
    r"""A flow that trains a natural language inference model"""

    config_path = Parameter('config',
    help = 'path to config file', default = "configs/train.json"
    )

    @step
    def start(self):
        r"""Start node"""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.next(self.init_system)

    @step
    def init_system(self):

        config = load_config(self.config_path)
        checkpoint_callback = ModelCheckpoint(
            dirpath = config.system.save_dir,
            verbose = True,
            save_top_k = 1,
            monitor = 'dev_loss',
            every_n_epochs = 1,
            mode = 'min')
        dm = ImdbDataModule(config)
        trainer = Trainer(
            logger = TensorBoardLogger(save_dir = LOG_DIR),
            max_epochs= config.system.optimizer.max_epochs,
            callbacks = [checkpoint_callback]
        )
    
        system = BertTextClassificationSystem(config, truncation = true)
        self.trainer = trainer
        self.dm = dm
        self.system = system

        self.next(self.train)

    @step
    def train(self):
        
        self.system.train()
        self.trainer.fit(self.system, self.dm)
        self.next(self.offline_test)

    @step
    def offline_test(self):
        r""" Offline `test` on the trainer. saves to a log file """
        self.system.eval()
        self.trainer.test(self.system, self.dm, ckpt_path = 'best')
        results = self.system.test_results

        print(results)
        log_file = join(LOG_DIR, 'train_flow', 'results.json')
        os.makedirs(os.path.dirname(log_file), exist_ok = True)
        to_json(results, log_file)
         
        self.next(self.end)

    @step    
    def end(self):
        """End node"""
        print('done! great work!')

if __name__ == "__main__":
    flow = TrainClassifier()