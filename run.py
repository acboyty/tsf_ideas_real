import numpy as np 
import torch
import torch.nn as nn
from data_loader import data_loader, data_loader_t1d, data_loader_idea
from sklearn.model_selection import train_test_split
from model import RNNModel, RNNGATEModel, IDEAModel
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import os


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)


DATA_DIR = 'data/prophet'
INPUT_TS = 'with_holiday'  # without_holiday, with_holiday

USE_HOLIDAY = 'idea'  # no_use, feature, gate, idea
GATE_VAR = 'meal'  # bolus, meal

LOOKBACK = 30
LOOKAHEAD = 1

MAX_EPOCH = 800

SEED = 1314
seed_everything(SEED)


if USE_HOLIDAY in {'no_use', 'feature', 'gate'}:
    X, y = data_loader(data_dir=DATA_DIR, lookback=LOOKBACK, lookahead=LOOKAHEAD, input_ts=INPUT_TS)
    # X, y = data_loader_t1d(data_dir=DATA_DIR, lookback=LOOKBACK, lookahead=6, gate_var=GATE_VAR)
    
    if 'shunfeng' in DATA_DIR:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
        print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
        # print(np.sum(X_test[:, :, 1]))

elif USE_HOLIDAY in {'idea'}:
    X, h, y = data_loader_idea(data_dir=DATA_DIR, lookback=LOOKBACK, lookahead=LOOKAHEAD, input_ts=INPUT_TS)

    if 'shunfeng' in DATA_DIR:
        X_train, X_test, h_train, h_test, y_train, y_test = train_test_split(X, h, y, test_size=0.2, shuffle=False)
        print(X_train.shape, X_test.shape, h_train.shape, h_test.shape, y_train.shape, y_test.shape)
    else:
        X_train, X_test, h_train, h_test, y_train, y_test = train_test_split(X, h, y, test_size=0.4, shuffle=False)
        X_val, X_test, h_val, h_test, y_val, y_test = train_test_split(X_test, h_test, y_test, test_size=0.5, shuffle=False)
        print(X_train.shape, X_val.shape, X_test.shape, 
            h_train.shape, h_val.shape, h_test.shape, 
            y_train.shape, y_val.shape, y_test.shape)


if USE_HOLIDAY == 'no_use':
    X_train[:, :, [1]] = 0
    # X_val[:, :, [1]] = 0
    X_test[:, :, [1]] = 0

    model = RNNModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, input_dim=2, hid_dim=40, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_test, y_val=y_test, 
        metric='mae', max_epoch=MAX_EPOCH, patience=3000000, 
        batch_size=128, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


elif USE_HOLIDAY == 'feature':
    model = RNNModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, input_dim=2, hid_dim=40, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_test, y_val=y_test, 
        metric='mae', max_epoch=MAX_EPOCH, patience=3000000, 
        batch_size=128, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


elif USE_HOLIDAY == 'gate':
    model = RNNGATEModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, hid_dim=40, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, y_train=y_train, 
        X_val=X_test, y_val=y_test, 
        metric='mae', max_epoch=MAX_EPOCH, patience=3000000, 
        batch_size=128, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test)


elif USE_HOLIDAY == 'idea':
    model = IDEAModel(
        lookback=LOOKBACK, lookahead=LOOKAHEAD, h_dim=1, hid_dim=40, 
        device='cuda', data_dir=DATA_DIR, task=USE_HOLIDAY
    )
    model.fit(
        X_train=X_train, h_train=h_train, y_train=y_train, 
        X_val=X_test, h_val=h_test, y_val=y_test, 
        metric='mae', max_epoch=MAX_EPOCH, patience=3000000, 
        batch_size=128, lr=1e-3, weight_decay=1e-3
    )
    forecast = model.eval(X_test=X_test, h_test=h_test)


print(f'MAE {mean_absolute_error(y_test, forecast):.5f}', end=' ')
print(f'MSE {mean_squared_error(y_test, forecast):.5f}')

np.save(os.path.join(DATA_DIR, 'y_test'), y_test)
np.save(os.path.join(DATA_DIR, USE_HOLIDAY + '.forecast'), forecast)
