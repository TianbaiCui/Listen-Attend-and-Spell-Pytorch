from six.moves import cPickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def get_data(data_table,i):
    return np.load(data_table.loc[i]['input'])

def load_dataset(data_path):
    data_table = pd.read_csv(data_path,index_col=0)
    #for i in tqdm(range(len(data_table))):
    #    X.append(np.load(data_table.loc[i]['input']))

    X = Parallel(n_jobs=-2,backend="threading")(delayed(get_data)(data_table,i) for i in tqdm(range(len(data_table))))

    Y = []
    for i in tqdm(range(len(data_table))):
        Y.append([int(v) for v in data_table.loc[i]['label'].split(' ')[1:]])
    return X,Y

# Input x: list of np array with shape (timestep,feature)
# Return new_x : a np array of shape (len(x), padded_timestep, feature)
def ZeroPadding(x,pad_len):
    features = x[0].shape[-1]
    new_x = np.zeros((len(x),pad_len,features))
    for idx,ins in enumerate(x):
        new_x[idx,:len(ins),:] = ins
    return new_x

# A transfer function for LAS label
# We need to collapse repeated label and make them onehot encoded
# each sequence should end with an <eos> (index = 1)
# Input y: list of np array with shape ()
# Output tuple: (indices, values, shape)
def OneHotEncode(Y,max_len,max_idx=30):
    new_y = np.zeros((len(Y),max_len,max_idx))
    for idx,label_seq in enumerate(Y):
        cnt = 0
        for label in label_seq:
            new_y[idx,cnt,label] = 1.0
            cnt += 1
            if cnt == max_len-1:
                break
        new_y[idx,cnt,1] = 1.0 # <eos>
    return new_y

class LibrispeechDataset(Dataset):
    def __init__(self, data_path, batch_size, max_label_len,bucketing,drop_last=False,training=False):
        print('Loading LibriSpeech data from',data_path,'...',flush=True)
        X,Y = load_dataset(data_path)
        if not bucketing:
            max_timestep = max([len(x) for x in X])
            self.X = ZeroPadding(X,max_timestep)
            self.Y = OneHotEncode(Y,max_label_len)
        else:
            print('Bucketing data ...',flush=True)
            bucket_x = []
            bucket_y = []
            for b in tqdm(range(int(np.ceil(len(X)/batch_size)))):
                left = b*batch_size
                if (b+1)*batch_size<len(X):
                    right = (b+1)*batch_size
                else:
                    if drop_last:
                        break
                    else:
                        right = len(X)
                pad_len = len(X[left]) if (len(X[left]) % 8) == 0 else\
                          len(X[left])+(8-len(X[left])%8)
                if training:
                    onehot_len = min(max([len(y) for y in Y[left:right]])+1,max_label_len)
                else:
                    onehot_len = max([len(y) for y in Y[left:right]])+1
                bucket_x.append(ZeroPadding(X[left:right], pad_len))
                bucket_y.append(OneHotEncode(Y[left:right], onehot_len))
            self.X = bucket_x
            self.Y = bucket_y
    def __getitem__(self, index):
        return self.X[index],self.Y[index]
    def __len__(self):
        return len(self.X)


def create_dataloader(data_path, max_label_len, batch_size, shuffle, bucketing, drop_last=False, training=False, 
                    **kwargs):
    if not bucketing:
        return DataLoader(LibrispeechDataset(data_path, batch_size,max_label_len,bucketing), 
                          batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,training=training)
    else:
        return DataLoader(LibrispeechDataset(data_path, batch_size,max_label_len,bucketing,drop_last=drop_last), 
                          batch_size=1,shuffle=shuffle,training=training)