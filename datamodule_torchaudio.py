import yaml
import torch
import torch.nn as nn
import numpy as np
from torchaudio import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

# Load config file for experiment

class DataModule(LightningDataModule):
    def __init__(self, data_path, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.config = conf
        self.trainset = datasets.LIBRISPEECH(data_path, url="train-clean-100", download=True)
        self.valset = datasets.LIBRISPEECH(data_path, url="test-clean", download=True)
        char_map = {
            "<PAD>": 0,
            '<SPACE>': 1,
            'a': 2,
            'b': 3,
            'c': 4,
            'd': 5,
            'e': 6,
            'f': 7,
            'g': 8,
            'h': 9,
            'i': 10,
            'j': 11,
            'k': 12,
            'l': 13,
            'm': 14,
            'n': 15,
            'o': 16,
            'p': 17,
            'q': 18,
            'r': 19,
            's': 20,
            't': 21,
            'u': 22,
            'v': 23,
            'w': 24,
            'x': 25,
            'y': 26,
            'z': 27,
            "'": 28,
            "<EOS>": 29,
        }
        index_map = {v: k for k, v in char_map.items()}
        index_map[1] = ' '
        self.char_map = char_map
        self.vocab_size = len(char_map.keys())
        self.index_map = index_map
        self.train_transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=40)
        self.val_transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=40)
        self.time_scale = 2**conf['model_parameter']['listener_layer']

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        int_sequence.append(self.char_map['<EOS>'])
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

    def OneHotEncode(self, Y,max_len):
        new_y = np.zeros((len(Y),max_len,self.vocab_size))
        for idx,label_seq in enumerate(Y):
            cnt = 0
            for label in label_seq:
                new_y[idx,cnt,label] = 1.0
                cnt += 1
                if cnt == max_len-1:
                    break
            new_y[idx,cnt,1] = 1.0 # <eos>
        return new_y

    def data_processing(self, data, data_type="train"):
        spectrograms = []
        labels = []
        #input_lengths = []
        #label_lengths = []
        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.train_transform(waveform).squeeze(0).transpose(0, 1)
            else:
                spec = self.val_transform(waveform).squeeze(0).transpose(0, 1)
            spectrograms.append(spec)
            label = torch.Tensor(self.text_to_int(utterance.lower()))
            labels.append(label)
            #input_lengths.append(spec.shape[0]//2)
            #label_lengths.append(len(label))       
        max_len = max([x.shape[0] for x in spectrograms])
        # max_label_len = max([len(y) for y in labels])
        #print(f"max_Len: {max_len}")
        #print(f'shape of spectrograms: {spectrograms[0].shape}')
        if max_len % self.time_scale != 0:
            max_len += (self.time_scale - max_len) % self.time_scale
        spectrograms[0] = nn.ConstantPad1d((0, max_len - spectrograms[0].shape[0]), 0.0)(spectrograms[0].transpose(0,1)).transpose(0,1)
        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)#.unsqueeze(1).transpose(2, 3)
        labels[0] = nn.ConstantPad1d((0, max_len - labels[0].shape[0]), 0.0)(labels[0])
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)#.unsqueeze(-1)
        labels = nn.functional.one_hot(labels.long(), num_classes=self.vocab_size)
        return spectrograms, labels #, input_lengths, label_lengths

    def train_dataloader(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size=self.config['training_parameter']['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=6,
            collate_fn=lambda x: self.data_processing(x, 'train')
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.valset,
            batch_size=self.config['training_parameter']['batch_size'],
            shuffle=False,  # collate_fn=CollateBatch,
            drop_last=False,
            pin_memory=True,
            num_workers=6,
            collate_fn=lambda x: self.data_processing(x, 'valid')
        )
        return val_loader
