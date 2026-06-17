# -*- coding: UTF-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def find_file(filename):
    """优先读取当前目录文件，若不存在则读取data目录下文件。"""
    if os.path.exists(filename):
        return filename

    data_path = os.path.join("data", filename)
    if os.path.exists(data_path):
        return data_path

    raise FileNotFoundError("未找到文件：" + filename)


def load_data(path):
    data = pd.read_csv(path, encoding="utf-8")

    unnamed_cols = [col for col in data.columns if "Unnamed" in col]
    if len(unnamed_cols) > 0:
        data = data.drop(columns=unnamed_cols)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return x, y


def get_score(model, x, y):
    pred_y = model.predict(x)

    if hasattr(model, "predict_proba"):
        score_y = model.predict_proba(x)[:, 1]
    else:
        score_y = pred_y

    accuracy = metrics.accuracy_score(y, pred_y)
    precision = metrics.precision_score(y, pred_y, zero_division=0)
    recall = metrics.recall_score(y, pred_y, zero_division=0)
    f1 = metrics.f1_score(y, pred_y, zero_division=0)
    auc = metrics.roc_auc_score(y, score_y)

    return accuracy, precision, recall, f1, auc


def print_score(model_name, dataset_name, model, x, y):
    accuracy, precision, recall, f1, auc = get_score(model, x, y)

    print(model_name, dataset_name)
    print("Accuracy: %.4f" % accuracy)
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1: %.4f" % f1)
    print("AUC: %.4f" % auc)
    print("-" * 60)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }


def main():
    train_path = find_file("train_preprocess.csv")
    test_path = find_file("test_preprocess.csv")

    x, y = load_data(train_path)
    test_x, test_y = load_data(test_path)

    train_x, valid_x, train_y, valid_y = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(
            solver="liblinear",
            penalty="l2",
            max_iter=1000,
            random_state=0
        ),
        "DecisionTree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=30,
            random_state=0
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            criterion="entropy",
            max_depth=10,
            min_samples_leaf=10,
            random_state=0
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=0
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "GaussianNB": GaussianNB()
    }

    results = []

    print("=" * 60)
    print("开始进行多模型对比实验")
    print("=" * 60)

    for model_name, model in models.items():
        print("当前模型：", model_name)
        print("初始参数：", model.get_params())
        print("-" * 60)

        model.fit(train_x, train_y)

        valid_result = print_score(
            model_name,
            "Validation",
            model,
            valid_x,
            valid_y
        )
        test_result = print_score(
            model_name,
            "Test",
            model,
            test_x,
            test_y
        )

        results.append(valid_result)
        results.append(test_result)

    result_df = pd.DataFrame(results)
    result_df.to_csv("model_compare_result.csv", index=False, encoding="utf_8_sig")

    print("=" * 60)
    print("模型对比结果已保存：model_compare_result.csv")
    print("=" * 60)
    print(result_df)

    print("=" * 60)
    print("开始对随机森林进行超参数调优")
    print("=" * 60)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_leaf": [5, 10, 20],
        "criterion": ["gini", "entropy"]
    }

    grid_model = GridSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=1
    )

    grid_model.fit(train_x, train_y)

    print("随机森林最优参数：")
    print(grid_model.best_params_)
    print("随机森林验证集最优F1：%.4f" % grid_model.best_score_)

    best_model = grid_model.best_estimator_

    print("=" * 60)
    print("调优后随机森林模型测试集结果")
    print("=" * 60)

    print_score(
        "TunedRandomForest",
        "Test",
        best_model,
        test_x,
        test_y
    )


if __name__ == "__main__":
    main()