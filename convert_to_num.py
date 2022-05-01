import datetime
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

class ConvertToNum:
    """
    工具性代码
    将字符串转化为数字，构建类似{字符1：数值1， 字符2：数值2}的编码词典，convert_to_num.pkl
    """
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'

    UNK = 0
    PAD = -1
    SOS = 1
    EOS = 2

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        self.fited = False

    def fit(self, sequence, min_count=1, max_count=None, max_feature=None):
        """
        fit数据进入词典
        :param sequence: [word1,word3,wordn..]
        :param min_count:最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """
        count = {}
        for a in sequence:
            if a not in count:
                count[a] = 0
            count[a] += 1

        # 词频限制
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        # 特征限制
        if isinstance(max_feature, int):
            count = dict(sorted(count.items(), key=lambda x: x[1])[:max_feature])
        for k in count:
            if k not in self.dict:
                self.dict[k] = len(self.dict)
        self.fited = True

    def transform(self, sentence, max_len=None, add_eos=False):
        """
        :param sentence: [word1,word3,wordn..]
        :return: [1,3,n..]
        """
        if max_len:
            r = [self.PAD]*max_len
        else:
            r = [self.PAD]*len(sentence)
        if max_len and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.dict.get(word, self.UNK)
        if add_eos:
            if r[-1] == self.PAD:
                pad_index = r.index(self.PAD)
                r[pad_index] = self.EOS
            else:
                r[-1] = self.EOS
        return r

    def inverse_transform(self, indices):
        """
        :param indices: [1,3,n..]
        :return: [word1,word3,wordn..]
        """
        self.inverse_dict = {v: k for k, v in self.dict.items()}
        sentence = []
        for i in indices:
            word = self.inverse_dict.get(i)
            if i != self.EOS:
                sentence.append(word)
            else:
                break
        return sentence

    def __len__(self):
        return len(self.dict)

if __name__ == "__main__":
    df1 = pd.read_table('../附件/附件1：估价训练数据.txt', header=None, sep='\t', parse_dates=[1, 11, 12, 32])
    df1.columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode', 'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear', 'displacement', 'gearbox', 'oiltype', 'newprice', 'anonymousFeature1', 'anonymousFeature2', 'anonymousFeature3', 'anonymousFeature4', 'anonymousFeature5', 'anonymousFeature6', 'anonymousFeature7', 'anonymousFeature8', 'anonymousFeature9', 'anonymousFeature10', 'anonymousFeature11', 'anonymousFeature12', 'anonymousFeature13', 'anonymousFeature14', 'anonymousFeature15', 'price']
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
    df1.replace("nan", np.nan, inplace=True)
    df1['tradeTime_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df1.tradeTime) / datetime.timedelta(1)
    df1['registerDate_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df1.registerDate) / datetime.timedelta(1)
    df1['licenseDate_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df1.licenseDate) / datetime.timedelta(1)
    df1['anonymousFeature7_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df1.anonymousFeature7) / datetime.timedelta(1)
    df1['anonymousFeature15_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df1.anonymousFeature15) / datetime.timedelta(1)

    df1['anonymousFeature7_q'] = df1.anonymousFeature7.dt.quarter
    # 季度
    df1['tradeTime_q'] = df1.tradeTime.dt.quarter
    df1['registerDate_q'] = df1.registerDate.dt.quarter
    df1['licenseDate_q'] = df1.licenseDate.dt.quarter
    df1['anonymousFeature13_m'] = df1.anonymousFeature13.dt.month
    # 缺失值填充
    data1 = df1
    df2 = pd.read_table('../附件/附件2：估价验证数据.txt', header=None, sep='\t', parse_dates=[1, 11, 12, 32])
    df2.columns = ['carid', 'tradeTime', 'brand', 'serial', 'model', 'mileage', 'color', 'cityId', 'carCode',
                   'transferCount', 'seatings', 'registerDate', 'licenseDate', 'country', 'maketype', 'modelyear',
                   'displacement', 'gearbox', 'oiltype', 'newprice', 'anonymousFeature1', 'anonymousFeature2',
                   'anonymousFeature3', 'anonymousFeature4', 'anonymousFeature5', 'anonymousFeature6',
                   'anonymousFeature7', 'anonymousFeature8', 'anonymousFeature9', 'anonymousFeature10',
                   'anonymousFeature11', 'anonymousFeature12', 'anonymousFeature13', 'anonymousFeature14',
                   'anonymousFeature15']
    df2['carid'] = df2['carid'].astype(str)
    df2['brand'] = df2['brand'].astype(str)
    df2['serial'] = df2['serial'].astype(str)
    df2['model'] = df2['model'].astype(str)
    df2['color'] = df2['color'].astype(str)
    df2['cityId'] = df2['cityId'].astype(str)
    df2['carCode'] = df2['carCode'].astype(str)
    df2['country'] = df2['country'].astype(str)
    df2['maketype'] = df2['maketype'].astype(str)
    df2['modelyear'] = df2['modelyear'].astype(str)
    df2['gearbox'] = df2['gearbox'].astype(str)
    df2['oiltype'] = df2['oiltype'].astype(str)
    df2['anonymousFeature13'] = pd.to_datetime(df2['anonymousFeature13'], format='%Y%m')
    df2['anonymousFeature7'] = pd.to_datetime(df2['anonymousFeature7'])
    df2['anonymousFeature15'] = pd.to_datetime(df2['anonymousFeature15'])
    df2.replace("nan", np.nan, inplace=True)
    df2['tradeTime_from_now'] = (pd.to_datetime(datetime.datetime.now().date()) - df2.tradeTime) / datetime.timedelta(1)
    df2['registerDate_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df2.registerDate) / datetime.timedelta(1)
    df2['licenseDate_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df2.licenseDate) / datetime.timedelta(1)
    df2['anonymousFeature7_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df1.anonymousFeature7) / datetime.timedelta(1)
    df2['anonymousFeature15_from_now'] = (pd.to_datetime(
        datetime.datetime.now().date()) - df2.anonymousFeature15) / datetime.timedelta(1)

    df2['anonymousFeature7_q'] = df1.anonymousFeature7.dt.quarter
    # 季度
    df2['tradeTime_q'] = df2.tradeTime.dt.quarter
    df2['registerDate_q'] = df2.registerDate.dt.quarter
    df2['licenseDate_q'] = df2.licenseDate.dt.quarter
    df2['anonymousFeature13_m'] = df2.anonymousFeature13.dt.month

    # 缺失值填充
    data2 = df2
    columns_to_normalize = ['tradeTime_from_now', 'registerDate_from_now', 'licenseDate_from_now', 'mileage',
                            'transferCount', 'seatings', 'displacement', 'newprice', 'anonymousFeature1',
                            'anonymousFeature2', 'anonymousFeature3', 'anonymousFeature4', 'anonymousFeature5',
                            'anonymousFeature6', 'anonymousFeature7_from_now', 'anonymousFeature8', 'anonymousFeature9',
                            'anonymousFeature10', 'anonymousFeature14', 'anonymousFeature15_from_now']
    cate_columns = ['brand', 'tradeTime_q', 'registerDate_q', 'licenseDate_q', 'anonymousFeature7_q',
                    'anonymousFeature13_m', 'serial', 'model', 'color', 'cityId', 'carCode', 'country', 'maketype',
                    'modelyear', 'gearbox', 'oiltype', 'anonymousFeature11', 'anonymousFeature12']

    data = data1.append(data2)
    data[cate_columns] = data[cate_columns].fillna('PAD')

    # 分类变量
    convert_to_num = ConvertToNum()
    bar = tqdm(data[cate_columns].iteritems(), total=len(data.columns))
    for idx, data in bar:
        convert_to_num.fit(list(data.values))

    pickle.dump(convert_to_num, open('./convert_to_num.pkl', 'wb'))
    print(convert_to_num.dict)
