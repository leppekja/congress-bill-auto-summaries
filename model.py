import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

SEED = 2932


def train_an_epoch(dataloader, optimizer):
    model.train()
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        probs = model(text)

        loss = loss_function(probs, label)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            print(f'Iteration: {idx}; Loss: {loss:.3f}.')


class Encoder(nn.Module):
    def __init__(self,
                 glove_vectors):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(glove_vectors)
