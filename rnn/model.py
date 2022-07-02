import torch
from torch import nn


class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HybridRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=0)
        pre_hidden = self.linear(combined)
        hidden = self.tanh(pre_hidden)
        output = self.linear_out(hidden)
        return hidden, output

    def init_hidden(self, minibatch_size):
        return torch.zeros(minibatch_size, self.hidden_size)
