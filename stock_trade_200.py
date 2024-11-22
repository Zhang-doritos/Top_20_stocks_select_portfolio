import pandas as pd
import numpy as np


class PortfolioManager:
    def __init__(self, total_investment, max_single_stock_ratio, transaction_cost_ratio, window_size):
        self.total_investment = total_investment  # 总初始投资金额
        self.max_single_stock_ratio = max_single_stock_ratio  # 单支股票最大投资比例
        self.transaction_cost_ratio = transaction_cost_ratio  # 交易成本比例
        self.window_size = window_size  # 窗口大小
        self.cash = total_investment  # 初始现金
        self.holdings = {}  # 持仓
        self.total_assets = []  # 总资产记录

    def _calculate_investments(self, cash, weights):
        investments = [
            min(cash * weight, self.total_investment * self.max_single_stock_ratio)
            for weight in weights
        ]
        total_investment_this_time = sum(investments)
        if total_investment_this_time > cash:
            factor = cash / total_investment_this_time
            investments = [investment * factor for investment in investments]
        return investments

    def _update_holdings(self, stock, price, investment):
        """根据投资金额调整持仓"""
        if stock not in self.holdings:
            self.holdings[stock] = 0

        current_holding = self.holdings[stock] * price
        if investment > current_holding:
            # 买入逻辑
            num_shares_to_buy = (investment - current_holding) / price
            self.cash -= investment  # 扣除投资金额
            self.cash -= investment * self.transaction_cost_ratio  # 扣除交易成本
            self.holdings[stock] += num_shares_to_buy
            return f"Stock {stock}: Bought {num_shares_to_buy:.2f} shares at ${price:.2f}."
        elif investment < current_holding:
            # 卖出逻辑
            num_shares_to_sell = (current_holding - investment) / price
            self.cash += (current_holding - investment)  # 增加现金
            self.cash -= (current_holding - investment) * self.transaction_cost_ratio  # 扣除交易成本
            self.holdings[stock] -= num_shares_to_sell
            return f"Stock {stock}: Sold {num_shares_to_sell:.2f} shares at ${price:.2f}."
        return None

    def _clear_unpredicted_holdings(self, predicted_stocks, current_prices):
        """清理不在预测中的股票"""
        transactions = []
        for stock in list(self.holdings.keys()):
            if stock not in predicted_stocks and self.holdings[stock] > 0:
                price = current_prices.get(stock, 0)
                if price > 0:
                    self.cash += self.holdings[stock] * price
                    self.cash -= self.holdings[stock] * price * self.transaction_cost_ratio
                    transactions.append(f"Stock {stock}: Sold all {self.holdings[stock]:.2f} shares at ${price:.2f} (not in prediction).")
                self.holdings[stock] = 0
        return transactions

    def trade(self, predicted_df, data):
        for step in range(len(predicted_df)):
            trade_time_idx = step * self.window_size + self.window_size
            if trade_time_idx >= len(data):
                print(f"Stopping: trade_time_idx {trade_time_idx} exceeds data range {len(data)}")
                break

            trade_time = data.index[trade_time_idx]
            predicted_stocks = [f"STOCK_{stock.split('_')[1]}_Close" for stock in predicted_df.iloc[step]]
            all_relevant_stocks = list(set(predicted_stocks + list(self.holdings.keys())))
            current_prices = data.loc[trade_time, all_relevant_stocks].dropna()

            missing_prices = [stock for stock in self.holdings if stock not in current_prices.index]
            if missing_prices:
                print(f"Warning: Missing prices for stocks in holdings: {missing_prices}")
                for stock in missing_prices:
                    current_prices[stock] = 0
                
            weights = np.geomspace(1, 0.01, len(predicted_stocks))
            weights /= np.sum(weights)
            weights = np.full(20, 0.05)  # 生成一个长度为20，值均为0.05的数组
            investments = self._calculate_investments(self.cash, weights)

            print(f"Transaction {step + 1} at time {trade_time}:")
            print("Stock transactions:")
            for stock, investment in zip(predicted_stocks, investments):
                if stock in current_prices.index:
                    transaction_msg = self._update_holdings(stock, current_prices[stock], investment)
                    if transaction_msg:
                        print(transaction_msg)

            transactions = self._clear_unpredicted_holdings(predicted_stocks, current_prices)
            for msg in transactions:
                print(msg)

            self.holdings = {stock: shares for stock, shares in self.holdings.items() if shares > 0}
            current_assets = self.cash + sum(
                self.holdings[stock] * current_prices[stock]
                for stock in self.holdings if stock in current_prices.index
            )
            self.total_assets.append(current_assets)
            print(f"Current assets: ${current_assets:.2f}\n")

        return self.total_assets

    def calculate_final_assets(self, data):
        final_time = data.index[-1]
        final_prices = data.loc[final_time]
        final_assets = self.cash + sum(
            self.holdings[stock] * final_prices[stock]
            for stock in self.holdings if stock in final_prices.index
        )
        return final_assets, self.holdings


# 数据加载
predicted_df = pd.read_csv('data/top_20_predicted-200.csv')
data = pd.read_csv('data/test_close_px.csv', index_col=0)
data = data.iloc[100:, :].copy()

# 配置参数
manager = PortfolioManager(
    total_investment=10000000,
    max_single_stock_ratio=0.1,
    transaction_cost_ratio=0.001,
    window_size=200
)

# 执行交易
total_assets = manager.trade(predicted_df, data)

# 计算最终资产
final_assets, final_holdings = manager.calculate_final_assets(data)

# 输出最终结果
print(f"Final assets: ${final_assets:.2f}")
print("Final holdings:")
for stock, shares in final_holdings.items():
    if shares > 0:
        print(f"Stock {stock}: {shares:.2f} shares")
