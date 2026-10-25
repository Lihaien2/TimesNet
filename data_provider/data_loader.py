import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from datasets import load_dataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
            max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df = ds_data["train"].to_pandas()
            test_df = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",
                                         repo_type="dataset")

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",
                                         repo_type="dataset")

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy", repo_type="dataset")
            test_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy", repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",
                                         repo_type="dataset")

            train_data = np.load(train_path)
            test_data = np.load(test_path)
            test_label = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                                  index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
            torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class Dataset_MultiStock(Dataset):
    """多股票面板数据集：按全市场日期切分，并按 target_end_date 清除跨界标签。"""

    # 仅允许事先定义的行情/技术指标进入模型，审计字段绝不作为特征。
    DEFAULT_FEATURES = (
        'open', 'high', 'low', 'close', 'volume', 'amount',
        'pct_chg', 'turn', 'ma5', 'ma20'
    )
    META_COLUMNS = {
        'date', 'target_start_date', 'target_end_date', 'stock_code', 'target',
        'is_constituent', 'is_st', 'historical_status_known',
        'is_blacklisted', 'is_delisted', 'is_eligible',
        'is_target_eligible', 'is_constituent_at_target'
    }

    def __init__(self, args, root_path, flag='train', size=None,
                 features='MS', data_path='panel.csv', target='target',
                 scale=True, timeenc=0, freq='d', seasonal_patterns=None,
                 stock_code_col='stock_code', window_stride=1,
                 train_ratio=0.7, vali_ratio=0.1):
        self.args = args
        self.seq_len, self.label_len, self.pred_len = size or (60, 0, 1)
        if flag not in ('train', 'val', 'test'):
            raise ValueError(f'不支持的 flag: {flag}')
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features, self.target, self.scale = features, target, scale
        self.timeenc, self.freq = timeenc, freq
        self.root_path, self.data_path = root_path, data_path
        self.stock_code_col, self.window_stride = stock_code_col, max(1, int(window_stride))
        self.train_ratio, self.vali_ratio = train_ratio, vali_ratio
        if self.label_len != 0:
            raise ValueError('Dataset_MultiStock 只支持 label_len=0')
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        required = {'date', self.stock_code_col, self.target}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'面板缺少必要列: {sorted(missing)}')
        if 'target_end_date' not in df.columns:
            raise ValueError('缺少 target_end_date；请重新生成面板，避免无法清除跨切分标签。')
        if 'target_start_date' not in df.columns:
            raise ValueError('缺少 target_start_date；请重新生成带入场日期的面板。')

        # 只用明确列出的行情特征，禁止将日期、标签或资格元数据泄漏给模型。
        self.feature_cols = [c for c in self.DEFAULT_FEATURES if c in df.columns]
        if not self.feature_cols:
            raise ValueError('未找到任何允许使用的行情特征列。')
        self.n_features = len(self.feature_cols)
        self.scale_cols = list(self.feature_cols)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['target_start_date'] = pd.to_datetime(df['target_start_date'], errors='coerce')
        df['target_end_date'] = pd.to_datetime(df['target_end_date'], errors='coerce')
        df = df.dropna(subset=['date', 'target_start_date', 'target_end_date', self.target])
        invalid_target_dates = (
                (df['target_start_date'] <= df['date'])
                | (df['target_end_date'] < df['target_start_date'])
        )
        if invalid_target_dates.any():
            example = df.loc[
                invalid_target_dates, [self.stock_code_col, 'date', 'target_start_date', 'target_end_date']].head(5)
            raise ValueError(
                '发现不合法的目标日期（要求 signal_date < target_start_date <= target_end_date）：\n' + example.to_string(
                    index=False))
        df = df.sort_values([self.stock_code_col, 'date']).reset_index(drop=True)
        dates = pd.Index(sorted(df['date'].unique()))
        if len(dates) < 3:
            raise ValueError('有效交易日期不足，无法进行时间切分。')
        # 固定按预测目标结束日期切分，边界均包含当天：
        # train: target_end_date <= 2024-01-31
        # val:   2024-01-31 < target_end_date <= 2024-08-31
        # test:  2024-08-31 < target_end_date <= 2025-12-31
        self.train_end = pd.Timestamp('2024-01-31')
        self.val_end = pd.Timestamp('2024-08-31')
        self.test_end = pd.Timestamp('2025-12-31')
        self.max_signal_date = pd.Timestamp(dates[-1])
        self.max_date = pd.Timestamp(self.test_end)
        print(
            f'[Dataset] 按目标结束日期切分: train≤{self.train_end.date()}, '
            f'val({self.train_end.date()}, {self.val_end.date()}], '
            f'test({self.val_end.date()}, {self.test_end.date()}]'
        )
        if df['target_end_date'].min() > self.train_end:
            raise ValueError('数据中没有落在训练目标结束日期范围内的记录。')

        # 统一的组合决策日期：基于全局交易日历，而不是每只股票各自的行号。
        # seq_len-1 保证决策日之前至少有 seq_len 个全局观测日期；单只股票仍需有足够历史行。
        self.decision_dates = pd.Index(dates[self.seq_len - 1::self.window_stride])
        self.decision_date_set = set(pd.Timestamp(d) for d in self.decision_dates)

        # 特征 scaler 仅使用训练时间区间内、标签未越过训练边界的原始记录拟合。
        # target scaler 则必须与模型实际看到的训练标签完全一致：同时应用
        # seq_len、全局 decision_dates/window_stride、标签结束日边界和 is_eligible 过滤。
        train_rows = df[df['target_end_date'] <= self.train_end]
        self.scaler = StandardScaler()
        if self.scale:
            if train_rows.empty:
                raise ValueError('训练时间段没有可用且标签结束于训练边界前的样本，无法拟合 scaler。')
            self.scaler.fit(train_rows[self.scale_cols].astype(float).values)

            actual_train_targets = []
            for _, train_g in df.groupby(self.stock_code_col, sort=False):
                train_g = train_g.sort_values('date').reset_index(drop=True)
                if len(train_g) < self.seq_len:
                    continue
                train_dates = train_g['date'].values
                train_ends = train_g['target_end_date'].values
                train_y = train_g[self.target].astype(float).values
                for train_i in range(self.seq_len - 1, len(train_g)):
                    signal_date = pd.Timestamp(train_dates[train_i])
                    if signal_date not in self.decision_date_set:
                        continue
                    if not (pd.Timestamp(train_ends[train_i]) <= self.train_end):
                        continue
                    if 'is_eligible' in train_g.columns:
                        eligible_value = train_g.loc[train_i, 'is_eligible']
                        if isinstance(eligible_value, str):
                            eligible = eligible_value.strip().lower() in ('true', '1', 'yes', 'y')
                        else:
                            eligible = bool(eligible_value) if pd.notna(eligible_value) else False
                        if not eligible:
                            continue
                    actual_train_targets.append(train_y[train_i])

            if not actual_train_targets:
                raise ValueError('按实际训练样本筛选后没有可用 target，无法拟合 target scaler。')
            t = np.asarray(actual_train_targets, dtype=np.float64)
            self.target_mean = float(np.mean(t))
            target_std = float(np.std(t, ddof=0))
            self.target_std = target_std if target_std > 0 else 1.0
            print(
                f'[Dataset] target scaler fitted on actual train labels: n={len(t)}, mean={self.target_mean:.8f}, std={self.target_std:.8f}')
        else:
            self.target_mean, self.target_std = None, None

        self.data_x, self.data_y, self.data_stamp, self.data_sid = [], [], [], []
        self.data_meta = []
        codes = sorted(df[self.stock_code_col].astype(str).unique())
        self.code_to_id = {c: i for i, c in enumerate(codes)}
        for code, g in df.groupby(self.stock_code_col, sort=False):
            g = g.sort_values('date').reset_index(drop=True)
            if len(g) < self.seq_len:
                continue
            raw_x = g[self.feature_cols].astype(float).values
            x_all = self.scaler.transform(raw_x) if self.scale else raw_x
            if self.timeenc == 0:
                stamps = np.zeros((len(g), 4), dtype=np.float32)
            else:
                stamps = time_features(pd.to_datetime(g['date'].values), freq=self.freq).transpose(1, 0).astype(
                    np.float32)
            dates_stock = g['date'].values
            ends_stock = g['target_end_date'].values
            y_all = g[self.target].astype(float).values
            # 输入可使用切分点之前的历史上下文；标签结束日必须留在对应分区内。
            for i in range(self.seq_len - 1, len(g)):
                signal_date = pd.Timestamp(dates_stock[i])
                # 所有股票仅在同一组全局决策日期产生样本。
                if signal_date not in self.decision_date_set:
                    continue
                target_end = pd.Timestamp(ends_stock[i])
                if self.set_type == 0:
                    keep = target_end <= self.train_end
                elif self.set_type == 1:
                    keep = self.train_end < target_end <= self.val_end
                else:
                    keep = self.val_end < target_end <= self.test_end
                if not keep:
                    continue
                # 若存在资格字段，训练/验证/测试样本都只取信号日可交易成分股。
                if 'is_eligible' in g.columns:
                    eligible_value = g.loc[i, 'is_eligible']
                    if isinstance(eligible_value, str):
                        eligible = eligible_value.strip().lower() in ('true', '1', 'yes', 'y')
                    else:
                        eligible = bool(eligible_value) if pd.notna(eligible_value) else False
                    if not eligible:
                        continue
                else:
                    eligible = True
                x = x_all[i - self.seq_len + 1:i + 1]
                y = y_all[i]
                if self.target_mean is not None:
                    y = (y - self.target_mean) / self.target_std
                self.data_x.append(x.astype(np.float32))
                self.data_y.append(np.array([y], dtype=np.float32))
                self.data_stamp.append((stamps[i - self.seq_len + 1:i + 1], stamps[i:i + 1]))
                self.data_sid.append(self.code_to_id[str(code)])
                self.data_meta.append({
                    'date': signal_date.strftime('%Y-%m-%d'),
                    'target_start_date': pd.Timestamp(g.loc[i, 'target_start_date']).strftime('%Y-%m-%d'),
                    'target_end_date': target_end.strftime('%Y-%m-%d'),
                    'stock_code': str(code),
                    'is_eligible': bool(eligible),
                })

        self.data_x = np.asarray(self.data_x, dtype=np.float32).reshape(-1, self.seq_len, self.n_features)
        self.data_y = np.asarray(self.data_y, dtype=np.float32).reshape(-1, 1)
        self.data_sid = np.asarray(self.data_sid, dtype=np.int64)
        if len(self.data_x) == 0:
            raise ValueError(
                f'{("train", "val", "test")[self.set_type]} 分区没有可用样本；检查日期边界、样本长度和资格过滤。')

        # ---- 审计：逐样本检查信号日期和标签结束日期是否落在本分区 ----
        _audit = pd.DataFrame(self.data_meta)
        _audit['date'] = pd.to_datetime(_audit['date'])
        _audit['target_start_date'] = pd.to_datetime(_audit['target_start_date'])
        _audit['target_end_date'] = pd.to_datetime(_audit['target_end_date'])
        _valid_target_dates = (
                (_audit['target_start_date'] > _audit['date'])
                & (_audit['target_end_date'] >= _audit['target_start_date'])
        )
        if self.set_type == 0:  # train: 目标结束日期截至 2024-01-31（含）
            _valid = _audit['target_end_date'] <= self.train_end
        elif self.set_type == 1:  # val: 目标结束日期在 2024-01-31 之后、2024-08-31（含）
            _valid = (
                    (_audit['target_end_date'] > self.train_end)
                    & (_audit['target_end_date'] <= self.val_end)
            )
        else:  # test: 目标结束日期在 2024-08-31 之后、2025-12-31（含）
            _valid = (
                    (_audit['target_end_date'] > self.val_end)
                    & (_audit['target_end_date'] <= self.test_end)
            )

        _valid = _valid & _valid_target_dates
        _bad = _audit.loc[~_valid, ['stock_code', 'date', 'target_start_date', 'target_end_date']]
        print(
            f'[Audit] split={("train", "val", "test")[self.set_type]}, '
            f'分区日期检查违规样本={len(_bad)}/{len(_audit)}'
        )
        if not _bad.empty:
            print('[Audit] 前 10 条违规记录：')
            print(_bad.head(10).to_string(index=False))
            raise ValueError('发现标签结束日期或信号日期跨越数据分区边界；已停止运行。')

        print(
            f'[Dataset] split={("train", "val", "test")[self.set_type]}, samples={len(self.data_x)}, features={self.feature_cols}')
        print(
            f'[Dataset] data_x mean={self.data_x.mean():.4f}, std={self.data_x.std():.4f}, max_abs={np.abs(self.data_x).max():.2f}')
        print(f'[Dataset] data_y mean={self.data_y.mean():.4f}, std={self.data_y.std():.4f}')

    def __getitem__(self, index):
        x, y = self.data_x[index], self.data_y[index]
        # Forecasting models expect y as [time, target_dim], not [target_dim].
        # With pred_len=1 and label_len=0, each sample must therefore be [1, 1].
        y = y.reshape(self.pred_len, -1)
        xm, ym = self.data_stamp[index]
        if self.set_type == 0:
            return x, y, xm, ym
        return x, y, xm, ym, self.data_sid[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        if self.target_mean is None:
            return data
        return data * self.target_std + self.target_mean

