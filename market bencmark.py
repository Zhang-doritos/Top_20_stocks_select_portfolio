import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 初始资金
INIT_FUND = 1e7

# 加载数据
stock_data = pd.read_csv("data/test_close_px.csv", index_col=0)
stock_data = stock_data.iloc[100:, :].copy()  # 跳过前100行数据

# 计算每只股票的 return
stock_returns = stock_data.iloc[-1] / stock_data.iloc[0] - 1  # 每只股票从起始到结束的 return

# 计算 Market Benchmark
average_price = stock_data.mean(axis=1)  # 每一天的股票平均价格
market_benchmark = (average_price / average_price.iloc[0]) * INIT_FUND  # 标准化并乘以初始资金
portfolio_return = (average_price.iloc[-1] / average_price.iloc[0]) - 1  # 总体平均 return

# 打印结果
print("每只股票的 Return：")
print(stock_returns)
print("\n市场基准平均 Return:")
print(portfolio_return)

# 可视化每一只股票的 Return
plt.figure(figsize=(12, 6))
plt.bar(stock_data.columns, stock_returns, color="skyblue", edgecolor="black")
plt.axhline(y=portfolio_return, color="red", linestyle="--", label="Market Benchmark Return")
plt.title("Individual Stock Returns and Market Benchmark")
plt.xlabel("Stocks")
plt.ylabel("Return")
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis="y")
plt.show()

# 可视化 Market Benchmark 随时间变化
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, market_benchmark, label="Market Benchmark", color="blue")
plt.title("Market Benchmark Over Time")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()
