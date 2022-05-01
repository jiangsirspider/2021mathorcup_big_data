import pandas as pd
import numpy as np
import datetime
import pickle
from convert_to_num import ConvertToNum
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# 变量缩写替换字典
my_dict = {'carid':'Carid','tradeTime':'trt', 'tradeTime_from_now':'tr_f_n', 'tradeTime_q':'trt_q', 'cityId': 'cid', 'transferCount':'transferCount', 'seatings':'seatings', 'registerDate':'rdate','registerDate_from_now':'rdate_f_n','registerDate_q':'rdate_q','licenseDate':'ldate', 'licenseDate_from_now':'ldate_f_n', 'licenseDate_q':'ldate_q', 'anonymousFeature1':'ayf1', 'anonymousFeature2':'ayf2', 'anonymousFeature3':'ayf3', 'anonymousFeature4':'ayf4', 'anonymousFeature5':'ayf5', 'anonymousFeature6':'ayf6', 'anonymousFeature7':'ayf7', 'anonymousFeature8':'ayf8', 'anonymousFeature9':'ayf9', 'anonymousFeature10':'ayf10', 'anonymousFeature11':'ayf11', 'anonymousFeature12':'ayf12', 'anonymousFeature13':'ayf13', 'anonymousFeature14':'ayf14', 'anonymousFeature15':'ayf15','anonymousFeature7_from_now':'ayf7_f_n', 'anonymousFeature7_q':'ayf7_q', 'anonymousFeature13_m':'ayf13_m', 'updatePriceTimeJson':'udpt_json', 'updatePriceCount':'udpc', 'updatePricePeriod':'udpp', 'updatePricePeriodFirst':'udppf', 'updatePriceRange':'udpr', 'updatePriceVariance':'udpv'}

# 设置中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main(df1, name):
    """
    主程序
    :return:
    """
    # 数据类型转换
    df1['carid'] = df1['carid'].astype(str)
    df1['brand'] = df1['brand'].astype(str)
    df1['serial'] = df1['serial'].astype(str)
    df1['model'] = df1['model'].astype(str)
    df1['color'] = df1['color'].astype(str)
    df1['cityId'] = df1['cityId'].astype(str)
    df1['carCode'] = df1['carCode'].astype(str)
    df1['country'] = df1['country'].astype(str)
    df1['maketype'] = df1['maketype'].astype(str)
    df1['modelyear'] = df1['modelyear'].astype(str)
    df1['gearbox'] = df1['gearbox'].astype(str)
    df1['oiltype'] = df1['oiltype'].astype(str)
    df1['anonymousFeature13'] = pd.to_datetime(df1['anonymousFeature13'], format='%Y%m')
    df1['anonymousFeature7'] = pd.to_datetime(df1['anonymousFeature7'])
    df1['anonymousFeature15'] = pd.to_datetime(df1['anonymousFeature15'])
    df1.replace("nan",np.nan,inplace=True)

    # 构建一些新变量，详见论文中的变量说明
    df1['tradeTime_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.tradeTime)/datetime.timedelta(1)
    df1['registerDate_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.registerDate)/datetime.timedelta(1)
    df1['licenseDate_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.licenseDate)/datetime.timedelta(1)
    df1['anonymousFeature7_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.anonymousFeature7)/datetime.timedelta(1)
    df1['anonymousFeature15_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.anonymousFeature15)/datetime.timedelta(1)
    df1['tradeTime_q'] = df1.tradeTime.dt.quarter
    df1['registerDate_q'] = df1.registerDate.dt.quarter
    df1['licenseDate_q'] = df1.licenseDate.dt.quarter
    df1['anonymousFeature7_q'] = df1.anonymousFeature7.dt.quarter
    df1['anonymousFeature13_m'] = df1.anonymousFeature13.dt.month

    # 转换索引列
    df1 = df1.reset_index()

    ############################################## 步骤：针对源数据采用随机森林预测进行填充 ###############################################
    print("############### step：针对源数据{}采用随机森林预测进行填充 ###################".format(name))

    # 等待预测的分类变量列表c_list， 数值型变量列表v_list
    c_list = ['anonymousFeature12','carCode', 'country', 'maketype', 'modelyear', 'gearbox', 'anonymousFeature7_q', 'anonymousFeature13_m', 'anonymousFeature11']
    v_list = ['anonymousFeature8']

    # 构建无缺失值的数据data1， columns_to_normalize为数值型变量，cate_columns为分类变量
    columns_to_normalize = ['tradeTime_from_now', 'registerDate_from_now', 'licenseDate_from_now', 'mileage', 'transferCount', 'seatings', 'displacement', 'newprice', 'anonymousFeature2', 'anonymousFeature3', 'anonymousFeature5', 'anonymousFeature6', 'anonymousFeature14']
    cate_columns = ['brand','tradeTime_q', 'registerDate_q', 'licenseDate_q', 'serial', 'model', 'color', 'cityId', 'oiltype']
    data_columns = columns_to_normalize + cate_columns
    data1 = df1[data_columns].dropna()

    # 将分类变量进行编码，转换为数值
    convert_to_num = pickle.load(open('./convert_to_num.pkl', 'rb'))
    data1[cate_columns] = data1[cate_columns].apply(convert_to_num.transform)
    data1 = data1.reset_index()  # 转换索引列

    def func(item, rf):
        """
        对缺失变量进行预测
        :param item: 待预测的变量
        :param rf: 模型
        :return:
        """
        df_train = df1[~df1[item].isna()][['index', item]].merge(right=data1, how='left', on='index')
        df_pred = df1[df1[item].isna()][['index', item]].merge(right=data1, how='left', on='index')
        if len(df_pred) == 0:
            print("sore of {}: {}".format(item, 1))
            return
        x_train, x_test, y_train, y_test = train_test_split(df_train[data_columns].values, df_train[item].values,   test_size=0.2)
        # 标准化
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
        # 训练模型
        model = rf.fit(x_train, y_train)
        print("sore of {}: {}".format(item, rf.score(x_test, y_test)))
        # 将预测的数据更新进入原表
        df_pred[item] = [i for i in rf.predict(df_pred[data_columns].values)]
        for idx, d in df_pred[['index', item]].iterrows():
            the_index = d['index']
            the_data = d[item]
            df1.loc[df1['index'] == the_index, item] = the_data
    # 对v_list中的变量进行预测
    for i in v_list:
        #创建随机森林分类器对象
        rf = RandomForestRegressor(random_state=0, n_estimators=100,  oob_score=True)  #袋外误差
        func(i, rf)  # df1[df1.carCode.isna()]
    print('==================================')
    # 对c_list中的变量进行预测
    for i in c_list:
        #创建随机森林分类器对象
        rf = RandomForestClassifier(random_state=0,   n_estimators=100,  oob_score=True)
        func(i, rf)

    # 随机森林得到特征值重要性
    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(data_columns, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    x_values = list(range(len(importances)))
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, [my_dict[i] if i in my_dict else i for i in data_columns], rotation='vertical')
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()

    # 数据保存
    pickle.dump(df1, open('./data/{}.pkl'.format(name), 'wb'))

print("#####################################################################")

if __name__ == "__main__":
    # 读取数据
    df1 = pd.read_table('../附件/附件1：估价训练数据.txt', header=None, sep='\t', parse_dates=[1, 11, 12, 32])  # 训练数据
    df1.columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode', 'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear', 'displacement', 'gearbox', 'oiltype', 'newprice', 'anonymousFeature1', 'anonymousFeature2', 'anonymousFeature3', 'anonymousFeature4', 'anonymousFeature5', 'anonymousFeature6','anonymousFeature7', 'anonymousFeature8', 'anonymousFeature9', 'anonymousFeature10', 'anonymousFeature11', 'anonymousFeature12', 'anonymousFeature13', 'anonymousFeature14', 'anonymousFeature15', 'price']
    df2 = pd.read_table('../附件/附件2：估价验证数据.txt', header=None, sep='\t', parse_dates=[1, 11, 12, 32])  # 验证数据
    df2.columns=['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode', 'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear', 'displacement', 'gearbox', 'oiltype', 'newprice', 'anonymousFeature1', 'anonymousFeature2', 'anonymousFeature3', 'anonymousFeature4', 'anonymousFeature5', 'anonymousFeature6', 'anonymousFeature7', 'anonymousFeature8', 'anonymousFeature9', 'anonymousFeature10', 'anonymousFeature11', 'anonymousFeature12', 'anonymousFeature13', 'anonymousFeature14', 'anonymousFeature15']
    # 对数据中的缺失值数据进行填充
    for i in [(df1, 'df1'), (df2, 'df2')]:
        main(*i)
