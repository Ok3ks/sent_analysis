from transformers import BertForSequenceClassification, BertTokenizerFast
from src.dataset import Newsgroup
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch 
from torch import optim
from os.path import join
import os
import torch.nn.functional as F

import numpy as np

class ImdbDataModule(pl.LightningDataModule):
    r"""Data Module around datasets"""
    def __init__(self, config):
        super().__init__()
        train_dataset = IMDB(split = 'train').
        train_dataset = train_dataset.shuffle(seed = 42)

        train_dataset = train_dataset.train_test_split(split = 0.7)
        dev_dataset = IMDB(split = 'train').train_test_split(split = 0.3)
        test_dataset = IMDB(split = 'test')

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.batch_size = config.system.optimizer.batch_size
        self.num_workers = config.system.optimizer.num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,
        shuffle = True, num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size= self.batch_size, 
        shuffle = True, num_workers= self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,
        shuffle = True, num_workers = self.num_workers)


class BertTextClassificationSystem(pl.LightningModule):
    "Pytorch Lightning system to classify documents"

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
    
    def get_model(self):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        model.train()
        return model

    def get_tokenizer(self):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        return tokenizer

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.config.system.optimizer.lr)
        return optimizer

    def _common_step(self, batch, _):
        #Modify here and replace with gpt-3 embeddings 
        input_ids = self.tokenizer(batch['text'], return_attention_mask= True, return_token_type_ids= False, truncation = True, return_tensors = 'pt', padding = True)
        labels = batch['label']
        print(labels.size)

        output = self.model(**input_ids)
        logits = output.logits
        logits = logits[:,0]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        with torch.no_grad():
            preds = torch.round(torch.sigmoid(logits))
            num_correct = torch.sum(preds == labels)
            num_total = labels.size(0)
            accuracy = num_correct/float(num_total)
        print(loss)
        return loss, accuracy

    def training_step(self, train_batch, batch_idx):
        loss,acc = self._common_step(train_batch, batch_idx)
        self.log_dict({'train_loss': loss, 'train_acc': acc},
        on_step = True, on_epoch = False, prog_bar = True , logger = True)
        return loss,acc

    def validation_step(self, dev_batch, batch_idx):
        loss, acc = self._common_step(dev_batch, batch_idx)
        return loss, acc

    def validation_epoch_end(self, outputs):
        avg_loss = np.mean(np.vstack(o[0] for o in outputs))
        avg_acc = np.mean(np.vstack(o[1] for o in outputs))
        self.log_dict({'dev_loss': avg_loss, 'dev_acc': avg_acc},
        on_step = False, on_epoch = True , prog_bar = True, logger = True)

    def test_step(self, test_batch, batch_idx):
        loss, acc = self._common_step(test_batch, batch_idx)
        return loss, acc

    def test_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(o[0] for o in outputs))
        avg_acc = torch.mean(torch.stack(o[1]) for o in outputs)

    def predict_step(self, batch, _):
        logits = self.model(avg_loss = torch.mean(torch.stack(o[0] for o in outputs)))
        probs = torch.sigmoid(logits)
        return probs

