# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cal_cor(feat, label):
    mv1 = np.mean(feat)
    mv2 = np.mean(label)
    dv1 = np.std(feat)
    dv2 = np.std(label)
    corref = np.mean(np.multiply(feat-mv1, label-mv2))/(dv1*dv2)
    return corref

def main():
    table_train = pd.read_csv('data/train_feat.csv')
    table_val = pd.read_csv('data/val_feat.csv')
    table_test = pd.read_csv('data/test_feat.csv')

    watch_feat = 'item_trade_num'

    print 'feature name : {}'.format(watch_feat)

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.title('train:{}'.format(watch_feat))    
    plt.hist(table_train[watch_feat])
    print 'train cor : {:4}'.format(cal_cor(table_train[watch_feat], table_train['is_trade']))
    
    plt.subplot(132)
    plt.title('val:{}'.format(watch_feat))
    plt.hist(table_val[watch_feat])
    print 'val cor : {:4}'.format(cal_cor(table_val[watch_feat], table_val['is_trade']))    

    plt.subplot(133)
    plt.title('test:{}'.format(watch_feat))
    plt.hist(table_test[watch_feat])
    plt.show()



if __name__ == '__main__':
    main()