import pandas as pd


class TradingStrategy:
    def __init__(
        self,
        strategy="momentum",
        lookback_period=20,
        stop_loss_threshold=0.02,
        rebalance_frequency="daily",
        initial_capital=100000,
    ):
        self.strategy = strategy
        self.lookback_period = lookback_period
        self.stop_loss_threshold = stop_loss_threshold
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital

    def calculate_momentum(self, prices):
        """Calculate momentum indicator"""
        returns = prices.pct_change()
        momentum = returns.rolling(window=self.lookback_period).mean()
        return momentum

    def generate_signals(self, prices):
        """Generate buy/sell signals based on momentum"""
        momentum = self.calculate_momentum(prices)

        # Buy signal: positive momentum
        # Sell signal: negative momentum or stop loss triggered
        signals = pd.DataFrame(index=prices.index)
        signals["signal"] = 0
        signals.loc[momentum > 0, "signal"] = 1  # Buy
        signals.loc[momentum < 0, "signal"] = -1  # Sell

        return signals

    def backtest(self, prices):
        """Run backtest on historical price data"""
        signals = self.generate_signals(prices)

        # Calculate portfolio value
        portfolio = self.initial_capital
        positions = []

        for i, signal in enumerate(signals["signal"]):
            if signal == 1 and not positions:
                # Buy
                positions.append(
                    {"entry_price": prices.iloc[i], "quantity": portfolio / prices.iloc[i]}
                )
                portfolio = 0
            elif signal == -1 and positions:
                # Sell
                for pos in positions:
                    profit = (prices.iloc[i] - pos["entry_price"]) * pos["quantity"]
                    portfolio += pos["entry_price"] * pos["quantity"] + profit
                positions = []

            # Check stop loss
            if positions:
                for pos in positions:
                    loss = (prices.iloc[i] - pos["entry_price"]) / pos["entry_price"]
                    if loss < -self.stop_loss_threshold:
                        portfolio += prices.iloc[i] * pos["quantity"]
                        positions = []
                        break

        return portfolio


# Example usage
if __name__ == "__main__":
    strategy = TradingStrategy(lookback_period=20, stop_loss_threshold=0.02)
    price_data = pd.read_csv("historical_prices.csv")
    final_value = strategy.backtest(price_data)
    print(f"Final portfolio value: ${final_value:,.2f}")
