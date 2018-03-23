#coding=utf-8
# pylint:disable=E1101
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

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