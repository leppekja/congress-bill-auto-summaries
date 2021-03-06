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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


def build_glove(vocab):
    # https://nlp.stanford.edu/projects/glove/
    VECTORS_CACHE_DIR = './.vector_cache'
    glove = GloVe(name='6B', cache=VECTORS_CACHE_DIR)
    glove_vectors = glove.get_vecs_by_tokens(vocab.itos)
    return glove_vectors


def train_an_epoch(model, dataloader, optimizer, loss_function):
    # loss_function = nn.CrossEntropyLoss().to(device)
    model.train()
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        output = model(label, text)
        output_dim = output.shape[-1]
        print('initial', output.size())
        print(label.size())
        output = output.view(-1, output_dim)
        target = label.view(-1)
        print(output.size())
        print(target.size())
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            print(f'Iteration: {idx}; Loss: {loss:.3f}.')


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 pretrained_embeddings,
                 device,
                 freeze_glove=False,
                 ):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # may have to adjust the padding_idx?
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze_glove)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size)

    def forward(self, text):
        # print('text: ', text.size())
        # Text size is batch size, bill length
        embedded = self.embedding(text).view(-1, len(text), 300)
        # Embedded size is sequence length, batch size, glove vecs size
        # print('embedded: ', embedded.size())
        outputs, (hidden, cell) = self.rnn(embedded)
        # print('outputs: ', outputs.size())
        # print('hidden: ', hidden.size())
        # print('cell: ', cell.size())
        # print('end of encoder')
        # each of these are size (1, batch_size, hidden_size)
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

        input_word = input_word.unsqueeze(0)
        # print('input: ', input_word.size())
        # print('hidden: ', hidden.size())
        # print('cell: ', cell.size())
        embedded = self.embedding(input_word)  # .view(len(input), 300, -1)
        # print('embedded: ', embedded.size())
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
        batch_size = label.shape[0]
        # print('batch size: ', batch_size)
        label_length = label.shape[1]
        # print('label_length: ', label_length)
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(label_length, batch_size,
                              vocab_size).to(self.device)

        hidden, cell = self.encoder(text)
        input_word = label[:, 0]
        # print('input word: ', input_word)
        # print(label.size())

        for t in range(1, label_length):
            output, hidden, cell = self.decoder(input_word, hidden, cell)
            # print('output size: ', output.size())
            # print('outputs: ', outputs.size())
            outputs[t] = output

            # pick next word
            top_choice = output.argmax(1)

            input_word = top_choice
        # print('outputs', outputs.size)
        # print(outputs)
        return outputs
