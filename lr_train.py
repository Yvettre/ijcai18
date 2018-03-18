#coding=utf-8
# pylint:disable=E1101
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

'''minmax'''
def scale(data, low=0, high=1):
    min_bycols = np.min(data, axis=0)
    max_bycols = np.max(data, axis=0)

    data -= np.tile(min_bycols, (len(data),1))
    data /= np.tile(max_bycols - min_bycols + 1e-15, (len(data),1))
    data *= (high-low)*np.ones(data.shape)

    return data

'''zscore'''
def scale_zscore(data, minus_mean_flag=True):
    mu_bycols = np.average(data, axis=0)
    std_bycols = np.std(data, axis=0)

    if minus_mean_flag:
        data -= np.tile(mu_bycols, (len(data), 1))
    data /= np.tile(std_bycols + 1e-15, (len(data), 1))

    return data

train_df = pd.read_csv('data/train_feat.csv')
val_df = pd.read_csv('data/val_feat.csv')
test_df = pd.read_csv('data/test_feat.csv')
label_train = train_df['is_trade'].values
label_val = val_df['is_trade'].values
train_df.fillna(0,inplace=True)
val_df.fillna(0,inplace=True)
train_df = train_df.apply(lambda x: (x - np.mean(x)) / np.std(x))
val_df = val_df.apply(lambda x: (x - np.mean(x)) / np.std(x))
data_train = train_df.drop(['instance_id', 'is_trade'], axis=1).values
data_val = val_df.drop(['instance_id', 'is_trade'], axis=1).values
id_test = test_df['instance_id']
data_test = test_df.drop(['instance_id'], axis=1).values


def main():
    model = LogisticRegression()
    model.fit(data_train, label_train)
    y_val = model.predict_proba(data_val)
    y_train = model.predict_proba(data_train)
    print log_loss(label_train, y_train)
    print log_loss(label_val, y_val)


if __name__ == '__main__':
    main() 