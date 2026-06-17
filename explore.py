#-*-coding:UTF-8-*-
from sklearn.utils import resample, shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置 matplotlib 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

train = pd.read_excel('data/data.xlsx')
test = pd.read_excel('data/eval.xlsx')

print(train.shape, test.shape)

print(train.describe())

print(test.describe())

train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index=True, sort=False)


nan_count = data.isnull().sum().sort_values(ascending=False)
nan_ratio = nan_count / len(data)

nan_data = pd.concat([nan_count, nan_ratio], axis=1, keys=['count', 'ratio'])

print(nan_data.head(10))

count = data.apply(lambda x: len(x.unique())).sort_values(ascending=False)
print(count.head(10))

train = data.loc[data['source'] == "train"]
test = data.loc[data['source'] == "test"]

train = train.drop(['source'], axis=1).copy()

skewness = train.apply(lambda x: x.skew())
skewness = skewness.sort_values(ascending=False)

# print(skewness.head(15))

skewness = skewness[abs(skewness) > 0.5]

plot_data = train[['购买力水平', '移动房车险数量']]

plot = plot_data.boxplot(column='购买力水平', by='移动房车险数量')

plot.set_xlabel('移动房车险数量')
plot.set_ylabel('购买力水平')

plt.show()


corr_target = train.corr()['移动房车险数量']

important_feature = corr_target[np.abs(corr_target) >= 0.01].index.tolist()

print(len(important_feature))

train = train[important_feature]

test = test[important_feature]


skewness = train.iloc[:, :-
                      1].apply(lambda x: x.skew()).sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]

train[skewness.index] = np.log1p(train[skewness.index])

skewness = test.iloc[:, :-
                     1].apply(lambda x: x.skew()).sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]

test[skewness.index] = np.log1p(test[skewness.index])

train.to_csv('train_preprocess.csv', encoding='utf_8_sig')
test.to_csv('test_preprocess.csv', encoding='utf_8_sig')


train['移动房车险数量'].value_counts()


train_up = train[train['移动房车险数量'] == 1]
train_down = train[train['移动房车险数量'] == 0]

train_up = resample(train_up, n_samples=696, random_state=0)
train_down = resample(train_down, n_samples=1095, random_state=0)

train = shuffle(pd.concat([train_up, train_down]))
