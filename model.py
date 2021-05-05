import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import GloVe
import numpy as np
import random

SEED = 2932
# https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb#scrollTo=1RidpPnfCVfI
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_glove(vocab):
    # https://nlp.stanford.edu/projects/glove/
    VECTORS_CACHE_DIR = './.vector_cache'
    glove = GloVe(name='6B', cache=VECTORS_CACHE_DIR)
    glove_vectors = glove.get_vecs_by_tokens(vocab.itos)
    return glove_vectors


def train_an_epoch(model, dataloader, optimizer):
    model.train()
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        probs = model(label, text)
        print(probs.shape)
        loss = loss_function(probs, label)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            print(f'Iteration: {idx}; Loss: {loss:.3f}.')


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 pretrained_embeddings,
                 freeze_glove=False,
                 batch_first=True,
                 ):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # may have to adjust the padding_idx?
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze_glove)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size)

    def forward(self, text):
        # returns size (batch, 300, input_size)
        embedded = self.embedding(text).view(len(text), 300, -1)

        outputs, (hidden, cell) = self.rnn(embedded)

        # each of these are size (1, 300, hidden_size)
        return hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, pretrained_embeddings, freeze_glove=False):
        super().__init__()

        # output dim should equal vocab size
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(300, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze_glove)

    def forward(self, input_word, hidden, cell):

        # input = input.unsqueeze(0)
        print(input_word.size())
        print(hidden.size())
        print(cell.size())
        embedded = self.embedding(input_word)  # .view(len(input), 300, -1)
        print(embedded.size())
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, label, text):
        batch_size = label.shape[1]
        label_length = label.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(label_length, batch_size, vocab_size)

        hidden, cell = self.encoder(text)

        input_word = label[0, :]

        for t in range(1, label_length):
            output, hidden, cell = self.decoder(input_word, hidden, cell)
            outputs[t] = output

            # pick next word
            top_choice = output.argmax(1)

            input_word = top_choice

        return outputs
