import yaml
import torch.utils.data as data
from pytorch_lightning import LightningDataModule

from util.librispeech_dataset import LibrispeechDataset

# Load config file for experiment

class DataModule(LightningDataModule):
    def __init__(self, config_path):
        super().__init__()
        print('Loading configure file at',config_path)

        with open(config_path, "r") as f:
            conf = yaml.load(f)

        self.trainset = LibrispeechDataset(conf['meta_variable']['data_path']+'/train.csv', **conf['model_parameter'], **conf['training_parameter'], training=True)
        self.valset = LibrispeechDataset(conf['meta_variable']['data_path']+'/dev.csv', **conf['model_parameter'], **conf['training_parameter'], drop_last=True)
        self.config = conf

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.trainset,
            batch_size=self.config['training_parameter']['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=6
        )
        return train_loader

    def val_dataloader(self):
        val_loader = data.DataLoader(
            self.valset,
            batch_size=self.config['training_parameter']['batch_size'],
            shuffle=False,  # collate_fn=CollateBatch,
            drop_last=True,
            pin_memory=True,
            num_workers=6,
        )
        return val_loader
