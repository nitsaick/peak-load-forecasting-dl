import datetime
import random

import numpy as np
import torch
from torch.utils import data


class TimeSeries(data.Dataset):
    def __init__(self, data_frame, input_time_interval, output_time_interval, output_keyword,
                 valid_rate=0.2, shuffle_seed=0):
        self.data_frame = data_frame
        self.data_channels = self.data_frame.head(1).values.shape[1]
        self.input_time_interval = input_time_interval
        self.output_time_interval = output_time_interval
        self.output_keyword = output_keyword
        self.get_data_list()
        self.dataset_size = len(self.inputs)
        self._split(valid_rate, shuffle_seed)

    def get_data_list(self):
        self.inputs = []
        self.targets = []

        head = self.data_frame.head(1).index[0]
        tail = self.data_frame.tail(1).index[0]
        data_head = head - datetime.timedelta(days=1)

        while True:
            data_head = data_head + datetime.timedelta(days=1)
            data_tail = data_head + datetime.timedelta(days=self.input_time_interval - 1)

            target_head = data_tail + datetime.timedelta(days=1)
            target_tail = target_head + datetime.timedelta(days=self.output_time_interval - 1)

            if target_tail > tail:
                break

            input = self.data_frame[data_head:data_tail]
            target = self.data_frame[target_head:target_tail][self.output_keyword]

            self.inputs.append(input)
            self.targets.append(target)

    def _split(self, valid_rate, shuffle_seed):
        self.indices = list(range(self.dataset_size))
        random.seed(shuffle_seed)
        random.shuffle(self.indices)
        split = int(np.floor((1 - valid_rate) * self.dataset_size))

        self.train_indices, self.valid_indices = self.indices[:split], self.indices[split:]
        self.train_dataset = data.Subset(self, self.train_indices)
        self.valid_dataset = data.Subset(self, self.valid_indices)

        self.train_sampler = data.RandomSampler(self.train_dataset)
        self.valid_sampler = data.SequentialSampler(self.valid_dataset)
        self.test_sampler = data.SequentialSampler(self)

    def get_dataloader(self, batch_size=1, num_workers=0):
        train_loader = data.DataLoader(self.train_dataset, batch_size=batch_size,
                                       sampler=self.train_sampler, num_workers=num_workers)
        valid_loader = data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                       sampler=self.valid_sampler, num_workers=num_workers)
        test_loader = data.DataLoader(self, batch_size=batch_size, sampler=self.test_sampler, num_workers=num_workers)
        return train_loader, valid_loader, test_loader

    def __getitem__(self, index):
        input = self.inputs[index].values.astype(np.float).transpose(1, 0)
        target = self.targets[index].values.astype(np.float)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target

    def __len__(self):
        return self.dataset_size
