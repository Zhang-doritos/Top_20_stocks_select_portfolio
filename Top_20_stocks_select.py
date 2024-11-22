#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized on Wed Nov 20 15:46:30 2024
"""

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# 加载已训练的模型
MODEL_FILENAME = "data/random_forest_model_200_200_model1.pkl"
DATA_FILENAME = "data/test_stock_sma_20.csv"
PREDICTED_CSV = "data/top_20_predicted-200.csv"
ACTUAL_CSV = "data/top_20_actual-200.csv"

model = load(MODEL_FILENAME)

# 加载数据
data = pd.read_csv(DATA_FILENAME, index_col=0).iloc[100:, :].copy()

# 定义函数
def extract_features(data, start_idx, window_size=200):
    """
    从输入数据中提取特征，滑动窗口计算收益率、波动率和总收益率。
    """
    window_data = data.iloc[start_idx:start_idx + window_size, :]
    returns = window_data.pct_change().dropna()  # 计算收益率
    feature = pd.DataFrame({
        "mean_return": returns.mean(),
        "volatility": returns.std(),
        "total_return": (window_data.iloc[-1] - window_data.iloc[0]) / window_data.iloc[0]
    }).T.values.flatten()  # 将特征展平为1D数组
    return feature

def calculate_actual_returns(data, start_idx, future_steps=200):
    """
    计算从 start_idx 开始未来若干步的收益率。
    """
    current_prices = data.iloc[start_idx, :]
    future_prices = data.iloc[start_idx + future_steps, :]
    returns = (future_prices - current_prices) / current_prices
    return returns.values.flatten()

# 参数设置
WINDOW_SIZE = 200  # 滑动窗口大小
FUTURE_STEPS = 200  # 预测未来步数
NUM_PREDICTIONS = len(data) // WINDOW_SIZE - 1  # 预测次数

predicted_returns = []
actual_returns = []

# 滑动窗口预测
for step in range(NUM_PREDICTIONS):
    start_idx = step * WINDOW_SIZE
    if start_idx + WINDOW_SIZE > len(data):
        break  # 防止超出数据范围
    
    # 提取特征
    features = extract_features(data, start_idx, window_size=WINDOW_SIZE)
    features = features.reshape(1, -1)  # 调整为模型输入格式

    # 预测未来收益率
    prediction = model.predict(features)
    predicted_returns.append(prediction.flatten())

    # 计算实际未来收益率
    actual_return = calculate_actual_returns(data, start_idx + WINDOW_SIZE, future_steps=FUTURE_STEPS)
    actual_returns.append(actual_return)

# 转换为 DataFrame
columns = [f"Stock_{i+1}" for i in range(data.shape[1])]
predicted_df = pd.DataFrame(np.array(predicted_returns), columns=columns)
actual_df = pd.DataFrame(np.array(actual_returns), columns=columns)

# 保存收益率最高的20只股票
top_20_predicted = []
top_20_actual = []

for i in range(predicted_df.shape[0]):
    top_20_predicted.append(predicted_df.iloc[i].nlargest(20).index.tolist())
    top_20_actual.append(actual_df.iloc[i].nlargest(20).index.tolist())

# 保存为CSV
pd.DataFrame(top_20_predicted, columns=[f"Predicted_{i+1}" for i in range(20)]).to_csv(PREDICTED_CSV, index=False)
pd.DataFrame(top_20_actual, columns=[f"Actual_{i+1}" for i in range(20)]).to_csv(ACTUAL_CSV, index=False)

# 计算重叠数量
overlap_count = [
    len(set(top_20_predicted[i]) & set(top_20_actual[i]))
    for i in range(len(top_20_predicted))
]

# 转换为 DataFrame
overlap_df = pd.DataFrame({
    "Prediction Step": list(range(1, len(overlap_count) + 1)),
    "Overlap Count": overlap_count
})

# 可视化重叠股票的数量
plt.figure(figsize=(12, 6))
plt.plot(overlap_df["Prediction Step"], overlap_df["Overlap Count"], marker='o', linestyle='-', label="Overlap Count")
plt.axhline(y=20, color='r', linestyle='--', label="Max Overlap (20)")
plt.xlabel("Prediction Step")
plt.ylabel("Number of Overlapping Stocks")
plt.title("Overlap Count Between Predicted and Actual Top 20 Stocks")
plt.legend()
plt.grid()
plt.show()
