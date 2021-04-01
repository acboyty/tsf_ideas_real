import numpy as np 
import os


def data_loader(data_dir, lookback=20, lookahead=1, input_ts='with_holiday'):
    if 'sin' in data_dir:
        without_holiday = np.load(os.path.join(data_dir, 'without_holiday.npy'))
        with_holiday = np.load(os.path.join(data_dir, 'with_holiday.npy'))
        holiday = np.load(os.path.join(data_dir, 'holiday.npy'))
        
        # gen batches
        X, y = [], []
        for st in range(without_holiday.shape[0]):
            if st + lookback + lookahead > without_holiday.shape[0]:
                break
            if input_ts == 'with_holiday':
                X.append(np.concatenate((with_holiday[st:st + lookback, np.newaxis], 
                    holiday[st:st + lookback, np.newaxis]), axis=1))
            elif input_ts == 'without_holiday':
                X.append(np.concatenate((without_holiday[st:st + lookback, np.newaxis], 
                    holiday[st:st + lookback, np.newaxis]), axis=1))
            y.append(with_holiday[st + lookback:st + lookback + lookahead])
        X, y = np.array(X), np.array(y)
    
    elif 'shunfeng' in data_dir:
        # delivery = np.load(os.path.join(data_dir, 'delivery.npy'))
        # holiday = np.load(os.path.join(data_dir, 'holiday.npy'))

        # X, y = [], []
        # for st in range(delivery.shape[1]):
        #     if st + lookback + lookahead > delivery.shape[1]:
        #         break
        #     X.append(np.concatenate((delivery[:200, st:st + lookback, np.newaxis], holiday[:200, st:st + lookback, np.newaxis]), axis=2))
        #     y.append(delivery[:200, st + lookback:st + lookback + lookahead])
        # X = np.concatenate(X, axis=0)
        # y = np.concatenate(y, axis=0)

        data = np.load(os.path.join(data_dir, 'data.npy'))[:100]

        X, y = [], []
        for st in range(data.shape[1]):
            if st + lookback + lookahead > data.shape[1]:
                break
            X.append(data[:, st:st + lookback])
            y.append(data[:, st + lookback:st + lookback + lookahead, 0])
        X, y = np.concatenate(X, axis=0), np.concatenate(y, axis=0)
    
    elif 'retail-data-analytics' in data_dir:
        data = np.load(os.path.join(data_dir, 'data.npy'))

        X, y = [], []
        for st in range(data.shape[1]):
            if st + lookback + lookahead > data.shape[1]:
                break
            X.append(data[:, st:st + lookback, :])
            y.append(data[:, st + lookback:st + lookback + lookahead, 0])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

    elif 'bike-sharing' in data_dir or 'NYC' in data_dir or 'london' in data_dir or 'prophet' in data_dir:
        data = np.load(os.path.join(data_dir, 'data.npy'))

        X, y = [], []
        for st in range(data.shape[0]):
            if st + lookback + lookahead > data.shape[0]:
                break
            X.append(data[st:st + lookback])
            y.append(data[st + lookback:st + lookback + lookahead, 0])
        X, y = np.array(X), np.array(y)

    return X, y

def data_loader_t1d(data_dir, lookback=24, lookahead=6, gate_var='bolus'):
    data = np.load(os.path.join(data_dir, 'data.npz'), allow_pickle=True)
    basetimes = data['basetimes']
    glucoses = data['glucoses']
    boluss = data['boluss']
    meals = data['meals']

    Xs = []
    ys = []

    for bt, gl, bl, ml in zip(basetimes, glucoses, boluss, meals):
        n = len(bt)

        for st in range(n):
            en = st + lookback + lookahead
            if en > n:
                break
            if gate_var == 'bolus':
                Xs.append(np.concatenate((gl[st:st + lookback, np.newaxis], bl[st:st + lookback, np.newaxis]), axis=1))
            elif gate_var == 'meal':
                Xs.append(np.concatenate((gl[st:st + lookback, np.newaxis], ml[st:st + lookback, np.newaxis]), axis=1))
            ys.append(gl[[st + lookback + lookahead - 1]])

    return np.array(Xs), np.array(ys)


def data_loader_idea(data_dir, lookback=20, lookahead=1, input_ts='with_holiday'):
    if 'sin' in data_dir:
        without_holiday = np.load(os.path.join(data_dir, 'without_holiday.npy'))
        with_holiday = np.load(os.path.join(data_dir, 'with_holiday.npy'))
        holiday = np.load(os.path.join(data_dir, 'holiday.npy'))
        
        # gen batches
        X, y = [], []
        for st in range(without_holiday.shape[0]):
            if st + lookback + lookahead > without_holiday.shape[0]:
                break
            if input_ts == 'with_holiday':
                X.append(np.concatenate((with_holiday[st:st + lookback, np.newaxis], 
                    holiday[st:st + lookback, np.newaxis]), axis=1))
            elif input_ts == 'without_holiday':
                X.append(np.concatenate((without_holiday[st:st + lookback, np.newaxis], 
                    holiday[st:st + lookback, np.newaxis]), axis=1))
            y.append(with_holiday[st + lookback:st + lookback + lookahead])
        X, y = np.array(X), np.array(y)
    
    elif 'shunfeng' in data_dir:
        data = np.load(os.path.join(data_dir, 'data.npy'))[:100]

        X, h, y = [], [], []
        for st in range(data.shape[1]):
            if st + lookback + lookahead > data.shape[1]:
                break
            X.append(data[:, st:st + lookback, [0]])
            h.append(data[:, st:st + lookback + lookahead, [1]])
            y.append(data[:, st + lookback:st + lookback + lookahead, 0])
        X, h, y = np.concatenate(X, axis=0), np.concatenate(h, axis=0), np.concatenate(y, axis=0)
    
    elif 'retail-data-analytics' in data_dir:
        data = np.load(os.path.join(data_dir, 'data.npy'))

        X, y = [], []
        for st in range(data.shape[1]):
            if st + lookback + lookahead > data.shape[1]:
                break
            X.append(data[:, st:st + lookback, :])
            y.append(data[:, st + lookback:st + lookback + lookahead, 0])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

    elif 'bike-sharing' in data_dir or 'NYC' in data_dir or 'london' in data_dir or 'prophet' in data_dir:
        data = np.load(os.path.join(data_dir, 'data.npy'))

        X, h, y = [], [], []
        for st in range(data.shape[0]):
            if st + lookback + lookahead > data.shape[0]:
                break
            X.append(data[st:st + lookback, [0]])
            h.append(data[st:st + lookback + lookahead, [1]])
            y.append(data[st + lookback:st + lookback + lookahead, 0])
        X, h, y = np.array(X), np.array(h), np.array(y)

    return X, h, y
    