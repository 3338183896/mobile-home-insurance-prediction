# -*- coding: UTF-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
def find_file(filename):
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
def evaluate_model(model, x, y):
    pred_y = model.predict(x)

    if hasattr(model, "predict_proba"):
        prob_y = model.predict_proba(x)[:, 1]
    else:
        prob_y = pred_y
    result = {
        "Accuracy": accuracy_score(y, pred_y),
        "Precision": precision_score(y, pred_y, zero_division=0),
        "Recall": recall_score(y, pred_y, zero_division=0),
        "F1": f1_score(y, pred_y, zero_division=0),
        "AUC": roc_auc_score(y, prob_y)
    }
    return result, pred_y, prob_y
def train_and_compare(train_x, train_y, test_x, test_y):
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
    fitted_models = {}
    print("=" * 60)
    print("开始进行多模型对比实验与结果可视化准备")
    print("=" * 60)
    for name, model in models.items():
        print("当前模型：", name)
        model.fit(train_x, train_y)
        fitted_models[name] = model
        test_result, pred_y, prob_y = evaluate_model(model, test_x, test_y)
        test_result["Model"] = name
        results.append(test_result)
        print("Accuracy: %.4f" % test_result["Accuracy"])
        print("Precision: %.4f" % test_result["Precision"])
        print("Recall: %.4f" % test_result["Recall"])
        print("F1: %.4f" % test_result["F1"])
        print("AUC: %.4f" % test_result["AUC"])
        print("-" * 60)
    result_df = pd.DataFrame(results)
    result_df = result_df[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
    return result_df, fitted_models
def tune_random_forest(train_x, train_y):
    print("=" * 60)
    print("开始对随机森林进行超参数调优")
    print("=" * 60)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    grid_model = GridSearchCV(
        estimator=RandomForestClassifier(random_state=0),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=1
    )
    grid_model.fit(train_x, train_y)
    print("随机森林最优参数：")
    print(grid_model.best_params_)
    print("随机森林交叉验证最优AUC：%.4f" % grid_model.best_score_)
    print("=" * 60)
    return grid_model.best_estimator_
def plot_model_compare(result_df, save_dir):
    plt.figure(figsize=(12, 7))
    models = result_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.18
    plt.bar(x - 1.5 * width, result_df["Accuracy"], width=width, label="Accuracy")
    plt.bar(x - 0.5 * width, result_df["Recall"], width=width, label="Recall")
    plt.bar(x + 0.5 * width, result_df["F1"], width=width, label="F1")
    plt.bar(x + 1.5 * width, result_df["AUC"], width=width, label="AUC")
    plt.xticks(x, models, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("指标值")
    plt.title("不同模型测试集性能对比图")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "01_模型性能对比图.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("已保存：", save_path)
def plot_conf_matrix(model, test_x, test_y, save_dir):
    pred_y = model.predict(test_x)
    cm = confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("随机森林混淆矩阵")
    plt.colorbar()
    classes = ["预测0", "预测1"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], ["真实0", "真实1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "02_混淆矩阵.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("已保存：", save_path)
def plot_roc_curve(models_dict, test_x, test_y, save_dir):
    plt.figure(figsize=(8, 6))
    selected_models = ["LogisticRegression", "RandomForest", "GaussianNB"]
    for model_name in selected_models:
        model = models_dict[model_name]
        if hasattr(model, "predict_proba"):
            prob_y = model.predict_proba(test_x)[:, 1]
        else:
            prob_y = model.predict(test_x)
        fpr, tpr, _ = roc_curve(test_y, prob_y)
        auc_value = roc_auc_score(test_y, prob_y)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name}(AUC={auc_value:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("假正例率FPR")
    plt.ylabel("真正例率TPR")
    plt.title("ROC曲线对比图")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "03_ROC曲线.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("已保存：", save_path)
def plot_feature_importance(model, feature_names, save_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        print("当前模型不支持特征重要性绘图")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = np.array(feature_names)[indices]
    top_importances = importances[indices]
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("重要性")
    plt.title("随机森林前15个重要特征图")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "04_特征重要性图.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print("已保存：", save_path)
def plot_technical_route(save_dir):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    steps = [
        (5, 11, "数据获取"),
        (5, 9.3, "数据清洗与预处理"),
        (5, 7.6, "特征工程与EDA"),
        (5, 5.9, "模型构建与训练"),
        (5, 4.2, "模型评估与优化"),
        (5, 2.5, "结果可视化与分析"),
        (5, 0.8, "总结与心得")
    ]
    for x, y, text in steps:
        rect = patches.FancyBboxPatch(
            (x - 1.8, y - 0.45),
            3.6,
            0.9,
            boxstyle="round,pad=0.02",
            edgecolor="black",
            facecolor="#d9ecff"
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=12)
    for i in range(len(steps) - 1):
        x1, y1, _ = steps[i]
        x2, y2, _ = steps[i + 1]
        ax.annotate(
            "",
            xy=(x2, y2 + 0.48),
            xytext=(x1, y1 - 0.48),
            arrowprops=dict(arrowstyle="->", lw=1.8)
        )
    ax.set_title("机器学习期末项目技术路线图", fontsize=16)
    save_path = os.path.join(save_dir, "05_技术路线图.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("已保存：", save_path)
def main():
    save_dir = "figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
    result_df, fitted_models = train_and_compare(train_x, train_y, test_x, test_y)
    result_df.to_csv(os.path.join(save_dir, "model_compare_result.csv"), index=False, encoding="utf_8_sig")
    print("模型对比结果：")
    print(result_df)
    print("=" * 60)
    best_rf = tune_random_forest(train_x, train_y)
    rf_result, rf_pred, rf_prob = evaluate_model(best_rf, test_x, test_y)
    print("调优后随机森林测试集结果：")
    print("Accuracy: %.4f" % rf_result["Accuracy"])
    print("Precision: %.4f" % rf_result["Precision"])
    print("Recall: %.4f" % rf_result["Recall"])
    print("F1: %.4f" % rf_result["F1"])
    print("AUC: %.4f" % rf_result["AUC"])
    print("=" * 60)
    plot_model_compare(result_df, save_dir)
    plot_conf_matrix(best_rf, test_x, test_y, save_dir)
    plot_roc_curve(fitted_models, test_x, test_y, save_dir)
    plot_feature_importance(best_rf, x.columns.tolist(), save_dir)
    plot_technical_route(save_dir)
    print("全部图表已生成，保存在 figures 文件夹中。")
if __name__ == "__main__":
    main()