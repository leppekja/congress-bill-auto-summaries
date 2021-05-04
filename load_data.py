import os
import torch
import pandas as pd
from numpy import floor
from numpy.random import shuffle
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe, Vocab
import torch.nn.functional as F
from collections import Counter
from functools import partial


def build_vocab(training_data, summary_col='summary', bill_col='bill'):
    '''
    Builds a Vocab object for a Dataset object.
    If default BillsDataset object, summary and bill are dict keys.
    '''
    # Here I used a single counter for both bills and summaries
    # Since we don't want to limit the vocab
    # Which seems right but I have no idea
    counter_words = Counter()

    for index in range(len(training_data)):
        example = training_data[index]
        counter_words.update(example[summary_col])
        counter_words.update(example[bill_col])

    return Vocab(counter_words)


def split(data, training_size, testing_size, valid_size, shuffle_data=True):
    ''' Takes in a pandas dataframe as data. Returns three BillsDataset objects'''
    assert training_size + testing_size + \
        valid_size == 1, 'Split sizes should sum to 1'

    def split_index(size, index_length):
        '''Converts decimal to # of samples to take'''
        return int(floor(size * index_length))
    # Split into training / testing / validation sets, assign as attributes.
    indices = list(range(len(data)))

    train_split = split_index(training_size, len(indices))
    test_split = split_index(testing_size, len(indices))

    if shuffle_data:
        shuffle(indices)

    training_data = data.iloc[indices[0:train_split]]
    test_data = data.iloc[indices[train_split:train_split + test_split]]
    validate_data = data.iloc[indices[train_split + test_split:]]

    return (BillsDataset(training_data, 'summary_clean', 'bill_clean'),
            BillsDataset(test_data, 'summary_clean', 'bill_clean'),
            BillsDataset(validate_data, 'summary_clean', 'bill_clean'))


def get_dataloaders(batch_size, vocab, max_summary_length, max_bill_length, **kwargs):
    '''
    kwargs for training_data:, test_data:, and validation_data:.
    Returns dict of dataloaders based on arg name input
    '''
    dataloaders = {}
    # Set params for the collate function
    # uses the max length of any bill and summary across the entire data set
    collate_fn = partial(
        collate_bills_fn, vocab=vocab, max_summary_length=max_summary_length, max_bill_length=max_bill_length)

    for dataset_name, data in kwargs.items():
        dataloaders[dataset_name] = DataLoader(
            data, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn)

    return dataloaders


def collate_bills_fn(batch, vocab, max_summary_length=512, max_bill_length=2048):
    '''
    Collates the batches into the dataloader. Pads unequal lengths with zeros
    based on the max lengths given.
    '''
    # https://nlp.stanford.edu/projects/glove/
    VECTORS_CACHE_DIR = './.vector_cache'
    glove = GloVe(name='6B', cache=VECTORS_CACHE_DIR)

    labels = []
    texts = []
    for idx, text_dict in enumerate(batch):
        # Get the label and the text
        label = text_dict['summary']
        text = text_dict['bill']
        # Translate to GloVe embeddings
        # label_vectors = glove.get_vecs_by_tokens(label)
        # text_vectors = glove.get_vecs_by_tokens(text)
        label_vectors = []
        text_vectors = []
        for word in label:
            label_vectors.append(vocab.stoi[word])
        for word in text:
            text_vectors.append(vocab.stoi[word])
        # Check lengths; see how many zero rows to pad
        label_length = len(label_vectors)
        text_length = len(text_vectors)
        labels_to_pad = max_summary_length - label_length
        text_to_pad = max_bill_length - text_length

        if label_length < max_summary_length:
            label_vectors.extend([0] * labels_to_pad)
            # label_vectors = F.pad(label_vectors, (0, 0, 0, labels_to_pad))
        if text_length < max_bill_length:
            text_vectors.extend([0] * text_to_pad)
            # text_vectors = F.pad(text_vectors, (0, 0, 0, text_to_pad))

        labels.append(torch.Tensor(label_vectors))
        texts.append(torch.Tensor(text_vectors))
    # Returns shape of (batch size, max_summary_length, embedding length) for each
    return (torch.stack(labels), torch.stack(texts))


class BillsDataset(Dataset):
    """
    Dataset for Congressional Bills
    Adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    and from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb.
    """

    def __init__(self, df, summaries_col, bills_col, transform=None):
        self.data = df.reset_index(drop=True)
        self.labels = summaries_col
        self.texts = bills_col
        self.transform = transform

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
