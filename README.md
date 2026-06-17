# 移动房车险购买预测

## 项目简介

本项目为机器学习实验课程综合实践项目，目标是根据客户人口属性、家庭结构、收入水平、购买力及已有保险购买情况，预测客户是否购买移动房车险。

项目完成了数据读取与清洗、特征选择、偏态特征变换、多模型训练与对比、随机森林超参数调优以及实验结果可视化等流程。

## 项目结构

```text
mobile-home-insurance-prediction/
├─ data/
│  ├─ data.xlsx
│  └─ eval.xlsx
├─ figures/
│  ├─ 01_模型性能对比图.png
│  ├─ 02_混淆矩阵.png
│  ├─ 03_ROC曲线.png
│  ├─ 04_特征重要性图.png
│  ├─ 05_技术路线图.png
│  └─ model_compare_result.csv
├─ explore.py
├─ model.py
├─ visualize.py
├─ train_preprocess.csv
├─ test_preprocess.csv
├─ model_compare_result.csv
└─ README.md
```

## 数据说明

原始训练集包含5822条样本和86个字段，独立测试集包含4000条样本和86个字段。预测目标为“移动房车险数量”，取值为0或1。

主要数据处理步骤如下：

1. 检查数据规模、描述性统计、缺失值和字段唯一值数量；
2. 根据特征与目标变量的相关系数进行特征筛选；
3. 对偏度绝对值大于0.5的数值特征进行`log1p`变换；
4. 生成预处理后的训练集和测试集；
5. 对多种分类模型进行训练、验证和测试。

## 使用的模型

项目对以下机器学习模型进行了对比：

* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* K-Nearest Neighbors
* Gaussian Naive Bayes

模型评价指标包括Accuracy、Precision、Recall、F1-score和AUC。项目还使用`GridSearchCV`对随机森林进行超参数搜索。

## 运行环境

推荐使用Python 3.8或更高版本。

安装依赖：

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

## 运行方法

进入项目根目录后，依次运行：

```bash
python explore.py
python model.py
python visualize.py
```

各程序作用如下：

* `explore.py`：完成数据探索、特征筛选和预处理数据导出；
* `model.py`：完成训练集与验证集划分、多模型对比和随机森林调参；
* `visualize.py`：生成模型性能对比图、混淆矩阵、ROC曲线、特征重要性图和技术路线图。

## 输出结果

运行程序后可得到：

* `train_preprocess.csv`
* `test_preprocess.csv`
* `model_compare_result.csv`
* `figures/01_模型性能对比图.png`
* `figures/02_混淆矩阵.png`
* `figures/03_ROC曲线.png`
* `figures/04_特征重要性图.png`
* `figures/05_技术路线图.png`

## 实验结论

实验结果表明，逻辑回归和随机森林具有较高的整体准确率，随机森林的AUC表现相对较好；高斯朴素贝叶斯具有较高的正类召回率，但精确率较低。

由于目标类别分布不均衡，仅依赖Accuracy不能全面评价模型，应结合Recall、F1-score和AUC进行综合分析。

后续可通过类别权重、SMOTE重采样、分类阈值调整和代价敏感学习等方法，进一步提高模型对少数类客户的识别能力。

## 注意事项

* 请从项目根目录运行程序，否则可能出现相对路径无法找到数据文件的问题。
* 若图表中文无法显示，请确认系统已安装黑体或微软雅黑字体。
* 数据文件仅用于课程实验与学习。
