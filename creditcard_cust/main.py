# This is a sample Python script.
import pandas as pd
import numpy as np
#import xgboost as xgb  放在这里import会出错
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import utils
import feature_eng


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def main():
    # 数据读取与特征构建
    print("loading data...")
    path = "./data/"
    info, actions, flag = utils.load_data(path)
    print("数据加载完成")
    # 合并数据和标签
    # data=pd.merge(flag,info,on='USRID',how='left')
    print('产出行为特征')
    C_feature = feature_eng.action_featrue(actions, flag.USRID)
    # C_feature=pd.read_csv('C_feature.csv')
    print('产出时间特征')
    T_feature = feature_eng.time_feature(actions, flag.USRID)
    # T_feature=pd.read_csv('T_feature.csv')
    print('产出模块特征')
    M_feature = feature_eng.module_feature(actions, flag.USRID)
    # M_feature=pd.read_csv('M_feature.csv')
    print('特征合并')
    feature = pd.merge(T_feature, M_feature, on=['USRID'], how='left')
    feature = pd.merge(feature, C_feature, on=['USRID'], how='left')
    # feature=pd.read_csv('featrue.csv')
    # 合并数据
    data = pd.merge(flag, info, on='USRID', how='left')
    data = pd.merge(data, feature, on='USRID', how='left')
    data = pd.read_csv(path + 'data.csv')
    print('数据合并完成\n')
    # 数据切分
    train,test=utils.data_split()

    # 取出特征列与标签
    train_usrid = train.pop('USRID')
    y = train.pop('FLAG')
    y = y.values
    col = list(train.columns)
    #X = train.values
    #print(X.shape)
    #print(len(col))
    test_usrid = test.pop('USRID')
    test_y = test.pop('FLAG')
    test = test.values
    # 建模与评估
    gbm = lgb.LGBMClassifier()
    gbm.fit(train, y)

    # 预估和产出结果文件
    res = pd.DataFrame()
    res['USRID'] = list(test_usrid.values)
    res['FLAG'] = gbm.predict_proba(test)[:, 1]
    res.to_csv(path + "baseline_res.csv")









# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("run credit card user predit ")
    main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
