import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


inputs = np.load('/home/covpreduser/Blob/v-tyan/tsf_ideas_real/data/shunfeng/inputs.npy')
# cnt = inputs.sum(axis=-1)
# idx = np.argsort(cnt)[::-1]
# inputs = inputs[idx[:100]]
date = np.load('/home/covpreduser/Blob/v-tyan/tsf_ideas_real/data/shunfeng/date.npy')

def gaussian(ins, is_training, mean=0, stddev=1e-2):
    stddev=1e-3
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        return ins + noise
    return ins

lookback = 30
lookahead = 5

USE_HOLIDAY = True

holidays = {
    '2019-04-05': 0,  # 清明
    '2020-04-04': 0,
    '2019-05-01': 1,  # 劳动节
    '2020-05-01': 1,
    '2019-06-07': 2,  # 端午节
    '2019-09-13': 3,  # 中秋节
    '2019-10-01': 4,  # 国庆节
    '2020-01-01': 5,  # 元旦
    '2020-01-25': 6,  # 春节
    '2019-06-18': 7,  # 618
    '2019-11-11': 8,  # 双11
}

H_DIM = np.max(list(holidays.values())) * 2 + 2
print('H DIM: ', H_DIM)


def parse_eh(d):
    th = 5
    eh = np.zeros(H_DIM)
    for h in holidays:
        g = (pd.Timestamp(str(d)) - pd.Timestamp(h)).days
        if abs(g) < th:
            idx = holidays[h] * 2 + (g > 0)
            eh[idx] = th - abs(g)
    return eh


def generate_data():
    L = inputs.shape[1]

    X = []
    y = []
    w = []
    h = []
    d = []

    for t in range(lookback, L - lookahead):
        X.append(inputs[:, t - lookback: t])
        y.append(inputs[:, t: t + lookahead])

        _w = np.array([pd.Timestamp(str(x)).weekday()
                       for x in date[t - lookback: t]])
        _w = np.expand_dims(_w, axis=0)
        _w = np.repeat(_w, inputs.shape[0], axis=0)
        w.append(_w)

        _h = [parse_eh(x) for x in date[t - lookback: t + lookahead]]
        _h = np.expand_dims(_h, axis=0)
        _h = np.repeat(_h, inputs.shape[0], axis=0) / 10.0
        h.append(_h)

        _d = date[t: t + lookahead]
        _d = [parse_eh(x).sum() > 0 for x in _d]
        _d = np.expand_dims(_d, axis=0)
        _d = np.repeat(_d, inputs.shape[0], axis=0)
        d.append(_d.astype('float'))

    X = torch.from_numpy(np.concatenate(X, axis=0))
    y = torch.from_numpy(np.concatenate(y, axis=0))
    w = torch.from_numpy(np.concatenate(w, axis=0))
    h = torch.from_numpy(np.concatenate(h, axis=0))
    d = np.concatenate(d, axis=0)

    return X, y, w, h, d


X, y, w, h, d = generate_data()

N = X.shape[0]
split = int(N * 0.8)


train_X, test_X = X[:split], X[split:]
train_y, test_y = y[:split], y[split:]
train_w, test_w = w[:split], w[split:]
train_h, test_h = h[:split], h[split:]
train_d, test_d = d[:split], d[split:]


########################### MODEL ###############################

mean = train_X.mean()
std = train_X.std()

# train_X = (train_X - mean) / std
# test_X = (test_X - mean) / std

# train_y = (train_y - mean) / std
# test_y = (test_y - mean) / std


train_data = TensorDataset(train_X, train_w, train_h, train_y)
test_data = TensorDataset(test_X, test_w, test_h, test_y)

train_loader = DataLoader(train_data, batch_size=32,
                          shuffle=True, num_workers=1, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=32,
                         shuffle=False, num_workers=1, pin_memory=True)

########################### MODEL ###############################


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self._build()

    def _build(self):
        self.week_em = nn.Embedding(7, 2)

        self.holiday_em = nn.Linear(H_DIM, 1)

        self.rnn = nn.GRU(2 + 1, 16)
        self.mlp = nn.Linear(16*2, lookahead)

    def forward(self, X, w, h):
        if not USE_HOLIDAY:
            w = w * 0
            h = h * 0

        w = self.week_em(w.long())
        X = X.float().unsqueeze(dim=-1)
        h = self.holiday_em(h.float())
        h = gaussian(h, self.training)

        h_back = h[:, :lookback]
        h_ahead = h[:, lookback:]

        X = X - h_back

        X = torch.cat([X, w], dim=-1)
        X = X.permute(1, 0, 2)

        hid, _ = self.rnn(X)

        hid = torch.cat(
            [
                hid.mean(dim=0),
                hid[-1]
            ], dim=-1
        )

        out = self.mlp(hid)
        out = out + h_ahead.squeeze(dim=-1)

        return out


def run(model, data_loader, optimizer):
    running_loss = 0.0

    pred = []
    label = []

    for data in data_loader:
        data = [x.cuda() for x in data]
        X, w, h, y = data

        p = model(X, w, h)

        pred.append(p)
        label.append(y)

        loss = F.l1_loss(y, p)
        running_loss += loss.item()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pred = torch.cat(pred, dim=0)
    label = torch.cat(label, dim=0)

    return running_loss / len(data_loader), pred, label

    
def evaluate_loss_details(pred, label, date):
    loss = F.l1_loss(pred, label, reduction='none')
    loss = loss.data.cpu().numpy()
    
    loss_h = np.sum(loss * date) / np.sum(date)
    loss_n = np.sum(loss * (1 - date)) / np.sum(1 - date)
    
    print('Holiday Loss: ', np.mean(loss_h))
    print('Non-Holiday Loss: ', np.mean(loss_n))


def train():
    model = Model()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for eid in range(50):
        print('-' * 12)
        print('On epoch {}'.format(eid))
        loss, pred, label = run(model, train_loader, optimizer)
        print('Train loss: {}'.format(loss))
        # evaluate_loss_details(pred, label, train_d)

        loss, pred, label = run(model, test_loader, optimizer=None)
        print('Val loss: {}'.format(loss))

        losses.append(loss)
        # evaluate_loss_details(pred, label, test_d)
    if USE_HOLIDAY:
        np.save('losses.idea.npy', losses)
        torch.save(pred.detach(), 'pred.idea.th')
        torch.save(label.detach(), 'label.idea.th')
    else:
        np.save('losses.nouse.npy', losses)
        torch.save(pred.detach(), 'pred.nouse.th')
        torch.save(label.detach(), 'label.nouse.th')


if __name__ == '__main__':
    train()
