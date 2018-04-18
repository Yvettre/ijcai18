#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

property_dict = {}
def count_up(instance):
    property_list_str = instance['item_property_list']
    property_list = property_list_str.split(';')
    for property_id in property_list:
        if not property_dict.has_key(property_id):
            property_dict[property_id] = [0, 0]# view_num, trade_num
        property_dict[property_id][0] += 1
        if instance['is_trade']:
            property_dict[property_id][1] += 1

def main():
    print 'reading...'
    train_df = pd.read_csv('data/round1_ijcai_18_train_20180301.txt', sep=' ')
    print 'applying...'
    train_df.apply(count_up, axis=1)
    property_df = pd.DataFrame(property_dict)
    property_df = property_df.T
    property_df.columns = ['view_num', 'trade_num']
    property_df['cvr'] = property_df['trade_num'] / property_df['view_num']
    property_df.sort_values(by='trade_num', ascending=False, inplace=True)
    property_df.to_csv('debug/property_debug.csv')
    print property_df[:10]


if __name__ == '__main__':
    main()