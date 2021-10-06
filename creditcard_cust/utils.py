# -*- coding: utf-8 -*-
"""
信用卡用户购买预测工具
"""

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from matplotlib.pylab import rcParams


def load_data(path):
    train_info=pd.read_csv(path+"train_info.csv",delimiter='\t')
    test_info=pd.read_csv(path+"test_info.csv",delimiter='\t')
    info=pd.concat([train_info,test_info])  # 默认做了行拼接
    train_action=pd.read_csv(path+"train_actions.csv",delimiter='\t')
    test_action=pd.read_csv(path+"test_actions.csv",delimiter='\t')
    actions=pd.concat([train_action,test_action])
    train_flag=pd.read_csv(path+"train_flag.csv",delimiter='\t')
    test_flag=pd.read_csv(path+"submit_example.csv",delimiter='\t')
    test_flag['FLAG']=-1
    flag=pd.concat([train_flag,test_flag])
    return info,actions,flag

def data_ana(data):
    data.describe()
    data['V2'].value_counts()  # 查看数据分布
    sns.distplot(data.loc[data['V1'] < 10, 'V1'])


def data_split():
    # 更多数据处理与train/test切分
    enc = LabelEncoder()
    onc = OneHotEncoder(sparse=False)
    for k in ['V2', 'V4', 'V5', 'APP']:
        try:
            data[k] = enc.fit_transform(data[k])
        except:
            data[k] = enc.fit_transform(data[k].map(int))
        s1 = onc.fit_transform(data[k].values.reshape(-1,1))
        # reshape(-1,1)：转换成1列，reshape(-1,2)：转换成2列，reshape(1,-1)：转换成1行 reshape(2,-1)：转换成2行
        s1 = pd.DataFrame(s1)
        s1.columns = ['k[0]' + '_' + str(i) for i in range(s1.shape[1])]
        s1['USRID'] = data['USRID'].values
        data = pd.merge(data, s1, on='USRID')
    data = data.drop(['APP', 'V2', 'V4', 'V5'], axis=1)
    train = data[data['FLAG'] != -1]
    test = data[data['FLAG'] == -1]
    return train,test
