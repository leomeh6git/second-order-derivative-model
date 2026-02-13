# === Import libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
import ta
import mplfinance as mpf
matplotlib.use('Qt5Agg')



# === Step 1: Download minute-level historical data (BTC as example) ===
# Binance 1-minute data can be downloaded via API, here using Yahoo as placeholder
data = pd.read_csv("btcusd.csv.gz", compression='gzip')

# Convert timestamp to datetime
data['Datetime'] = pd.to_datetime(data['timestamp'], unit='s')
data.set_index('Datetime', inplace=True)

# Rename columns to match your existing code
data.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}, inplace=True)

# Optional: filter for a smaller date range for faster testing
data = data.loc["2023-11-01":"2023-12-05"]

# Drop any missing rows
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(data.head())
# Reset index for easier handling
data.reset_index(inplace=True)

# === Step 2: Indicators and smoothing ===
# Smoothing for derivatives
data['Close_smooth'] = data['Close'].rolling(window=10, min_periods=1).mean()

# First and second derivatives
data['first_derivative'] = data['Close_smooth'].diff()
data['second_derivative'] = data['first_derivative'].diff()

# Momentum indicators
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
macd = ta.trend.MACD(data['Close'], window_slow=26, window_fast=12, window_sign=9)
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()

# === Step 3: Swing detection ===
swings = []
min_price_move = 0.002  # 0.2% min move for swing
min_duration = 3  # minimum 3-min sustained derivative

for i in range(2, len(data)):
    # Swing low: concavity change negative -> positive
    if data['second_derivative'].iloc[i-2] < 0 and data['second_derivative'].iloc[i-1] > 0:
        # Ensure price moved enough from previous swing high/low
        if len(swings) == 0 or abs(data['Close'].iloc[i-1] - swings[-1]['price'])/swings[-1]['price'] >= min_price_move:
            swings.append({'index': i-1, 'type': 'low', 'price': data['Close'].iloc[i-1]})
    # Swing high: concavity change positive -> negative
    elif data['second_derivative'].iloc[i-2] > 0 and data['second_derivative'].iloc[i-1] < 0:
        if len(swings) == 0 or abs(data['Close'].iloc[i-1] - swings[-1]['price'])/swings[-1]['price'] >= min_price_move:
            swings.append({'index': i-1, 'type': 'high', 'price': data['Close'].iloc[i-1]})

# === Step 4: Fibonacci retracements ===
def fibonacci_levels(swing_high, swing_low):
    diff = swing_high - swing_low
    return {
        'level_0': swing_low,
        'level_236': swing_low + 0.236*diff,
        'level_382': swing_low + 0.382*diff,
        'level_5': swing_low + 0.5*diff,
        'level_618': swing_low + 0.618*diff,
        'level_786': swing_low + 0.786*diff,
        'level_1': swing_high
    }

# === Step 5: Backtesting logic ===
trades = []

for i in range(len(data)):
    past_swings = [s for s in swings if s['index'] < i]
    if not past_swings:
        continue
    last_swing = past_swings[-1]
    if last_swing['type'] == 'low':
        swing_low = last_swing['price']
        highs = [s['price'] for s in past_swings if s['type'] == 'high']
        if not highs:
            continue
        swing_high = highs[-1]
        fib = fibonacci_levels(swing_high, swing_low)
        price = data['Close'].iloc[i]
        # Check golden zone
        if fib['level_618'] <= price <= fib['level_786']:
            # RSI + MACD confirmation
            if data['RSI'].iloc[i] > 50 and data['MACD'].iloc[i] > data['MACD_signal'].iloc[i]:
                trades.append({
                    'entry_index': i,
                    'entry_price': price,
                    'stop_loss': fib['level_382'],  # 30% Fib
                    'take_profit': swing_high,
                    'closed': False,
                    'exit_index': None,
                    'exit_price': None,
                    'pnl': None
                })

# === Step 6: Simple exit logic ===
for trade in trades:
    for j in range(trade['entry_index'], len(data)):
        price = data['Close'].iloc[j]
        if price <= trade['stop_loss'] or price >= trade['take_profit']:
            trade['exit_index'] = j
            trade['exit_price'] = price
            trade['pnl'] = trade['exit_price'] - trade['entry_price']
            trade['closed'] = True
            break
    if not trade['closed']:
        # Close at last price
        trade['exit_index'] = len(data)-1
        trade['exit_price'] = data['Close'].iloc[-1]
        trade['pnl'] = trade['exit_price'] - trade['entry_price']
        trade['closed'] = True

# === Step 7: Performance metrics ===
pnls = [t['pnl'] for t in trades]
win_rate = sum([1 for t in trades if t['pnl']>0])/len(trades) if trades else 0
total_return = sum(pnls)
print(f"Trades: {len(trades)}, Win rate: {win_rate:.2f}, Total return: {total_return:.2f}")

# === Step 8: Plot results ===
entry_indices = [t['entry_index'] for t in trades]
entry_prices = [t['entry_price'] for t in trades]
stop_prices = [t['stop_loss'] for t in trades]
tp_prices = [t['take_profit'] for t in trades]


apds = [
    mpf.make_addplot(entry_prices, type='scatter', markersize=80, marker='^', color='g'),
    mpf.make_addplot(stop_prices, type='scatter', markersize=50, marker='v', color='r'),
    mpf.make_addplot(tp_prices, type='scatter', markersize=50, marker='o', color='b')
]

mpf.plot(data.set_index('Datetime'), type='candle', volume=True,
         style='charles',
         title='Minute-level Fibonacci Golden Zone Entries',
         mav=(10,),
         addplot=apds,
         show_nontrading=False)

plt.scatter(entry_indices, entry_prices, marker='^', color='green', s=80, label='Entry')
plt.scatter(entry_indices, stop_prices, marker='v', color='red', s=50, label='Stop Loss')
plt.scatter(entry_indices, tp_prices, marker='o', color='blue', s=50, label='Take Profit')
plt.legend()
plt.show()

# === Adjusted run_strategy with capital and leverage ===
def run_strategy(df, capital_per_trade=100000, leverage=10):
    trades = []

    swings = []
    min_price_move = 0.002
    for i in range(2, len(df)):
        if df['second_derivative'].iloc[i-2] < 0 and df['second_derivative'].iloc[i-1] > 0:
            if len(swings) == 0 or abs(df['Close'].iloc[i-1] - swings[-1]['price'])/swings[-1]['price'] >= min_price_move:
                swings.append({'index': i-1, 'type': 'low', 'price': df['Close'].iloc[i-1]})
        elif df['second_derivative'].iloc[i-2] > 0 and df['second_derivative'].iloc[i-1] < 0:
            if len(swings) == 0 or abs(df['Close'].iloc[i-1] - swings[-1]['price'])/swings[-1]['price'] >= min_price_move:
                swings.append({'index': i-1, 'type': 'high', 'price': df['Close'].iloc[i-1]})

    for i in range(len(df)):
        past_swings = [s for s in swings if s['index'] < i]
        if not past_swings: continue
        last_swing = past_swings[-1]
        if last_swing['type'] == 'low':
            swing_low = last_swing['price']
            highs = [s['price'] for s in past_swings if s['type']=='high']
            if not highs: continue
            swing_high = highs[-1]
            fib = fibonacci_levels(swing_high, swing_low)
            price = df['Close'].iloc[i]
            if fib['level_618'] <= price <= fib['level_786']:
                if df['RSI'].iloc[i] > 50 and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                    # Compute position size
                    trade_units = (capital_per_trade * leverage) / price
                    trades.append({
                        'entry_index': i,
                        'entry_price': price,
                        'stop_loss': fib['level_382'],
                        'take_profit': swing_high,
                        'closed': False,
                        'exit_index': None,
                        'exit_price': None,
                        'pnl': None,
                        'units': trade_units
                    })

    # Close trades
    for trade in trades:
        for j in range(trade['entry_index'], len(df)):
            price = df['Close'].iloc[j]
            if price <= trade['stop_loss'] or price >= trade['take_profit']:
                trade['exit_index'] = j
                trade['exit_price'] = price
                trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['units']
                trade['closed'] = True
                break
        if not trade['closed']:
            trade['exit_index'] = len(df)-1
            trade['exit_price'] = df['Close'].iloc[-1]
            trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['units']
            trade['closed'] = True
    return trades


# === Adjusted Monte Carlo with fewer sims for speed ===
def monte_carlo_alpha(df, n_sim=200, capital_per_trade=100000, leverage=10):
    # Real PnL
    real_trades = run_strategy(df, capital_per_trade, leverage)
    real_pnl = sum([t['pnl'] for t in real_trades])

    # Log returns
    returns = np.log(df['Close']).diff().dropna().values
    simulated_pnls = []

    for _ in range(n_sim):
        sim_returns = np.random.choice(returns, size=len(returns), replace=True)
        sim_prices = df['Close'].iloc[0] * np.exp(np.cumsum(sim_returns))
        sim_df = df.copy()
        sim_df['Close'] = sim_prices

        sim_trades = run_strategy(sim_df, capital_per_trade, leverage)
        sim_pnl = sum([t['pnl'] for t in sim_trades])
        simulated_pnls.append(sim_pnl)

    simulated_mean = np.mean(simulated_pnls)
    alpha = real_pnl - simulated_mean

    print(f"Real PnL: ${real_pnl:,.2f}")
    print(f"Mean simulated PnL: ${simulated_mean:,.2f}")
    print(f"Estimated alpha: ${alpha:,.2f}")

    return alpha, simulated_pnls

# Run
alpha, sim_results = monte_carlo_alpha(data)

