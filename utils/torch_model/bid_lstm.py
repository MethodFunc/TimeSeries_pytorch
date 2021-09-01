from ..header import *


class BidLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers, batch_size):
        super().__init__()
        self.input = input_dim
        self.output = output_size
        self.layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_size

        self.fc = self.make_fc()

        self.lstm = nn.LSTM(self.input, self.hidden_dim // 2, self.layers, batch_first=True, bidirectional=True,
                            dropout=0.3)

    def make_fc(self):
        layers = [nn.BatchNorm1d(self.hidden_dim),
                  nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                  nn.ReLU(),
                  nn.Linear(self.hidden_dim // 4, self.output)]

        reg = nn.Sequential(*layers)

        return reg

    def forward(self, x):
        h0 = Variable(torch.zeros(self.layers * 2, x.size(0), self.hidden_dim // 2, device=device))
        c0 = Variable(torch.zeros(self.layers * 2, x.size(0), self.hidden_dim // 2, device=device))
        lstm_out, _ = self.lstm(x, (h0, c0))

        y_pred = self.fc(lstm_out[-1].view(self.batch_size, -1))

        return y_pred
