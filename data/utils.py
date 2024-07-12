import numpy as np
from sklearn.model_selection import TimeSeriesSplit

### time series cross validation
def split_tscv(df_data, kfold):
    tscv = TimeSeriesSplit(n_splits=kfold )
    train_index_list = []
    vali_index_list = []
    test_index_list = []
    
    flag1 = int(0.7*len(df_data))
    flag2 = int(len(df_data)-flag1)
    train_vali_data = df_data[0:flag1]
    test_data = df_data[-flag2:]
    
    test_list = test_data.index.tolist()
    test_index_list = [test_list for i in range(kfold) ]
    
    for i, (train_index, vali_index) in enumerate(tscv.split(train_vali_data)):
        train_index_list.append(train_index)
        vali_index_list.append(vali_index)
    
    return train_index_list, vali_index_list, test_index_list


### sliding window
def split_sequence(sequence, n_past, n_future):  
    X, y = list(), list()
    for i in range(0, len(sequence), 1):
        end_ix = i + n_past
        out_end_ix = end_ix + n_future
        if out_end_ix > len(sequence):
            break
        
        ### the traget column is the last column in the datafarm, starts with 0
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix,-1]             ### multi step
    
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)