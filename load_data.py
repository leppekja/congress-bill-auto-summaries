import os
import torch
import pandas as pd
from numpy import floor
from numpy.random import shuffle
from torch.utils.data import Dataset, DataLoader


class BillsDataset(Dataset):
    """
    Dataset for Congressional Bills
    Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    and from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb.
    """

    def __init__(self, csv, summaries_col, bills_col, transform=None):
        self.data = pd.read_csv(csv)
        self.labels = summaries_col
        self.texts = bills_col
        self.transform = transform
        self.training_data = None
        self.test_data = None
        self.validate_data = None

    def split(self, training_size, testing_size, valid_size, shuffle_data=True):
        assert training_size + testing_size + \
            valid_size == 1, 'Split sizes should sum to 1'

        def split_index(size, index_length):
            '''Converts decimal to # of samples to take'''
            return int(floor(size * index_length))
        # Split into training / testing / validation sets, assign as attributes.
        indices = list(range(len(self.data)))

        train_split = split_index(training_size, len(indices))
        test_split = split_index(testing_size, len(indices))

        if shuffle_data:
            shuffle(indices)

        self.training_data = self.data.iloc[indices[0:train_split]]
        self.test_data = self.data.iloc[indices[train_split:train_split + test_split]]
        self.validate_data = self.data.iloc[indices[train_split + test_split:]]

        return None

    def get_dataloaders(self, batch_size, collate_fn=None):

        train_dataloader = DataLoader(
            self.training_data, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn)

        test_dataloader = DataLoader(
            self.test_data, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn)

        validate_dataloader = DataLoader(
            self.validate_data, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn)

        return (train_dataloader, test_dataloader, validate_dataloader)

    def collate_fn(self, data):
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        summary = self.data.loc[idx, self.labels]
        bill = self.data.loc[idx, self.texts]

        sample = {'summary': summary, 'bill': bill}

        if self.transform:
            sample = self.transform(sample)

        return sample
