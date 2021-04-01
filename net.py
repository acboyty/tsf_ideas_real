import torch 
import torch.nn as nn


class RNNNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, seq_len):
        super(RNNNet, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        self._build()

    def _build(self):
        self.rnn = nn.GRU(self.input_dim, self.hid_dim)
        self.lin = nn.Linear(self.hid_dim, self.output_dim)

        self.ar = nn.Linear(self.seq_len, self.output_dim)

    def forward(self, x):
        # x [batch_size, seq_len, dim]

        _, hn = self.rnn(x.permute(1, 0, 2))
        hid = hn.squeeze(dim=0)  # hid [batch_size, dim]
        # TODO: the effect of pooling is not as good as hn
        # hid = torch.mean(output, dim=0) 
        out = self.lin(hid)  # out [batch_size, dim]

        # Auto Regression
        ar_out = self.ar(x[:, :, 0])

        out = out + ar_out

        return out


class RNNGATENet(nn.Module):
    def __init__(self, seq_len, hid_dim, output_dim):
        super(RNNGATENet, self).__init__()

        self.seq_len = seq_len
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self._build()

    def _build(self):
        self.rnn = RNNNet(2, self.hid_dim, self.output_dim, self.seq_len)
        # self.gate = nn.Sequential(
        #     nn.Linear(self.seq_len, self.seq_len), 
        #     nn.Linear(self.seq_len, self.output_dim)
        # )
        self.gate = nn.Linear(self.seq_len, self.output_dim)
        # self.gate2 = nn.Linear(self.seq_len, self.output_dim)

        # self.ar = nn.Linear(self.seq_len, self.output_dim)

    def forward(self, x):
        # x [batch_size, seq_len, 1]
        # h [batch_size, seq_len, holiday_dim]

        gate = self.gate(x[:, :, 1])
        # gate2 = self.gate2(x[:, :, 1])
        # if self.training == False:
        #     torch.save(gate, './holiday_effect/gate.th')
        #     torch.save(self.gate.weight, './holiday_effect/gate.weight.th')

        inputs = x.clone()
        inputs[:, :, 1] = 0

        rnn_out = self.rnn(inputs)

        # out = rnn_out * gate # + gate2
        out = rnn_out + gate
        # TODO: add tanh brings some optimization problem
        # out = rnn_out * torch.sigmoid(gate) * 2

        # # Auto Regression
        # ar_out = self.ar(x[:, :, 0])

        # out = out + ar_out

        return out


class IDEANet(nn.Module):
    def __init__(self, lookback, lookahead, h_dim, hid_dim):
        super(IDEANet, self).__init__()

        self.lookback = lookback
        self.lookahead = lookahead
        self.h_dim = h_dim
        self.hid_dim = hid_dim

        self._build()
    
    def _build(self):
        self.rnn = RNNNet(1, self.hid_dim, self.lookahead, self.lookback)
        self.holiday = nn.Linear(self.lookback + self.lookahead, self.lookback + self.lookahead)
        # self.holiday = nn.Sequential(
        #     nn.Linear(self.lookback + self.lookahead, self.lookback + self.lookahead), 
        #     nn.Linear(self.lookback + self.lookahead, self.lookback + self.lookahead)
        # )
        # TODO: MLP & Multi-holiday

    def forward(self, x, h):
        # x [batch_size, lookback, 1]
        # h [batch_size, lookback + lookahead, holiday_dim]

        holiday = self.holiday(h[:, :, 0])
        x_regular = x - holiday[:, :self.lookback].unsqueeze(dim=2)
        if self.training == False:
            torch.save(x, 'x.th')
            torch.save(x_regular, 'x_regular.th')

        rnn_out = self.rnn(x_regular)

        out = rnn_out + holiday[:, self.lookback:]

        return out