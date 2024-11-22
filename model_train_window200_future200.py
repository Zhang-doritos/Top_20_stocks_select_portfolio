#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:56:59 2024
防止过拟合1
    
    model1 = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,               # 限制树深度
            min_samples_split=15,       # 增加分裂所需的样本数
            min_samples_leaf=5,         # 增加叶子节点的样本数
            max_features="log2",        # 降低每次分裂使用的特征数
            random_state=42,
            n_jobs=-1
            )    



1470 1470 1470

Mean Squared Error (MSE): 5.572736157918906e-07
Correlation between predicted and actual values: 1.0000

随机森林的策略构建包含下列步骤：

获取数据：100只股票。

特征和标签提取：计算3个因子作为样本特征；计算未来200日的个股收益作为样本的标签。

特征预处理：进行缺失值处理。

模型训练与预测：使用随机森林模型进行训练和预测。

策略回测：利用2019到2022年数据进行训练，预测2022到2023年的股票表现。每日买入预测排名最靠前的20只股票，至少持有30日（1000个时间步），同时淘汰排名靠后的股票。具体而言，预测排名越靠前，分配到的资金越多且最大资金占用比例不超过90%；初始5日平均分配资金，之后，尽量使用剩余资金（这里设置最多用等量的1.5倍）。

模型评价：查看特征重要性和模型回归结果。

@author: doritos
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib 

# 加载数据
stock_data = pd.read_csv("/Users/doritos/Documents/ELEC7089/forestregressor/stock_sma_1000.csv", index_col=0)
stock_data = stock_data.iloc[1000:, :].copy()
assert stock_data.shape == (58483, 100), "数据维度与预期不符，请检查数据格式！"

# Step 1: 提取特征和标签
def calculate_features_and_labels(stock_data, window_size=200, future_steps=200):
    """
    计算特征和标签
    """
    features = []
    labels = []

    for i in range(window_size, len(stock_data) - future_steps):
        past_prices = stock_data.iloc[i - window_size:i, :]  # 过去window_size步的价格
        future_prices = stock_data.iloc[i:i + future_steps, :]  # 未来future_steps步的价格
        returns = past_prices.pct_change().dropna()  # 计算过去收益率

        # 计算特征
        feature = pd.DataFrame({
            "mean_return": returns.mean(),
            "volatility": returns.std(),
            "total_return": (past_prices.iloc[-1] - past_prices.iloc[0]) / past_prices.iloc[0]
        }).T.values.flatten()  # 展平为1D特征
        features.append(feature)

        # 计算标签：未来5步的收益率
        label = (future_prices.iloc[-1] - past_prices.iloc[-1]) / past_prices.iloc[-1]
        labels.append(label.values.flatten())

    return np.array(features), np.array(labels)

features, labels = calculate_features_and_labels(stock_data)

# Step 2: 数据预处理
def preprocess_data(features, labels):
    """
    对因子和标签进行预处理：缺失值填充 + 标准化
    """
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels

processed_features, processed_labels = preprocess_data(features, labels)

# Step 3: 模型训练与预测
X_train, X_test, y_train, y_test = train_test_split(processed_features, processed_labels, test_size=0.2, random_state=42)

# 检查是否已存在保存的模型
model_filename = "data/random_forest_model_200_200_model1.pkl"

try:
    # 加载已保存的模型
    model = joblib.load(model_filename)
    print(f"Loaded model from {model_filename}")
except FileNotFoundError:
    # 如果模型不存在，则训练新模型
    print("Model not found. Training a new model...")
    model = RandomForestRegressor(
            n_estimators=30,
            max_depth=8,               # 限制树深度
            min_samples_split=15,       # 增加分裂所需的样本数
            min_samples_leaf=5,         # 增加叶子节点的样本数
            max_features="log2",        # 降低每次分裂使用的特征数
            random_state=42,
            n_jobs=-1
            )    
    model.fit(X_train, y_train)

    # 保存模型到文件
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
predicted_returns = model.predict(X_test)


from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# 计算均方误差 (MSE)
mse = mean_squared_error(y_test.flatten(), predicted_returns.flatten())

# 计算预测值与实际值的相关性 (Correlation)
correlation, _ = pearsonr(y_test.flatten(), predicted_returns.flatten())

# 输出结果
print(f"Mean Squared Error (MSE): {mse}")
print(f"Correlation between predicted and actual values: {correlation:.4f}")

def visualize_predictions_vs_actuals(predicted, actual):
    """
    可视化预测值与实际值之间的关系
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actual.flatten(), predicted.flatten(), alpha=0.6)
    plt.title("Predicted vs Actual Values")
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.grid()
    plt.show()

# 调用函数
visualize_predictions_vs_actuals(predicted_returns, y_test)
