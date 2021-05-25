import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, embeddings):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embeddings = nn.Embedding.from_pretrained(embeddings)

        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers)
        self.dropout = nn.Dropout(.8)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, text, hidden):
        embedded = self.embeddings(text)
        output, hidden, cell = self.lstm(embedded, hidden)
        output = output[:, -1]
        output = self.dropout(output)
        logits = self.fc_out(output)

        return logits, hidden

    def init_state(self, sequence_length):
        return torch.zeros(self.num_layers, sequence_length, self.hidden_size, device=device)
