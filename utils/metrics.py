import numpy as np


# ============ 原有函数（保持不变） ============
def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, eps=1e-3):
    """收益率空间专用：加 eps 避免除零爆炸"""
    return np.mean(np.abs((true - pred) / (np.abs(true) + eps)))


def MSPE(pred, true, eps=1e-3):
    return np.mean(np.square((true - pred) / (np.abs(true) + eps)))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


# ============ 路径1+2 专用函数 ============
def compute_direction_accuracy_grouped(preds, trues, stock_ids,
                                        neutral_threshold=0.002):
    """
    按股票分组计算方向准确率（收益率空间）。
    preds/trues: 收益率预测值/真实值
    neutral_threshold: |true| 小于此值的样本不计入（避免把"横盘"算错）
    """
    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    stock_ids = np.asarray(stock_ids)

    correct_list = []
    for sid in np.unique(stock_ids):
        mask = stock_ids == sid
        p = preds[mask]
        t = trues[mask]

        nz = np.abs(t) > neutral_threshold
        if nz.sum() == 0:
            continue
        correct = (np.sign(p[nz]) == np.sign(t[nz])).astype(float)
        correct_list.append(correct)

    if not correct_list:
        return 0.0
    return float(np.concatenate(correct_list).mean())


def backtest_strategy_grouped(preds, trues, stock_ids,
                               signal_threshold=0.005,
                               cost_per_trade=0.0025,
                               risk_free_rate=0.0,
                               trading_days=252,
                               pred_horizon=10,
                               decision_interval=20):
    """
    按股票分组、非重叠持有期的等权组合回测。

    参数:
      pred_horizon      : 预测周期（target 是 close.shift(-N) 的收益率），
                          表示每笔交易的持仓天数
      decision_interval : 相邻两次决策的间隔（= Dataset 的 window_stride）
                          表示每 window_stride 天做一次决策
      trading_days      : 一年交易日数（默认 252）

    约束:
      decision_interval >= pred_horizon
      （否则相邻两笔交易的持仓重叠，回测结果无效）

    年化因子:
      periods_per_year = trading_days / decision_interval

    样例:
      pred_horizon=10, decision_interval=20:
        - 每笔交易持有 10 天
        - 每隔 20 天决策一次
        - 10 天持仓 + 10 天空仓
        - 年化因子 = 252 / 20 = 12.6
    """
    assert decision_interval >= pred_horizon, \
        f"decision_interval ({decision_interval}) 必须 >= " \
        f"pred_horizon ({pred_horizon})，否则持仓重叠"

    preds = np.asarray(preds).flatten()
    trues = np.asarray(trues).flatten()
    stock_ids = np.asarray(stock_ids)

    per_stock_returns = {}

    for sid in np.unique(stock_ids):
        mask = stock_ids == sid
        # 直接用所有样本（decision_interval >= pred_horizon 保证非重叠）
        p = preds[mask]
        t = trues[mask]

        if len(p) < 2:
            continue

        signals = np.zeros_like(p)
        signals[p > signal_threshold] = 1.0
        signals[p < -signal_threshold] = -1.0

        strategy_returns = signals * t - cost_per_trade * np.abs(signals)
        strategy_returns = strategy_returns[np.isfinite(strategy_returns)]
        strategy_returns = np.clip(strategy_returns, -0.5, 0.5)

        if len(strategy_returns) < 2:
            continue
        per_stock_returns[sid] = strategy_returns

    empty_ret = {k: 0.0 for k in
                 ['cumulative_return', 'annual_return', 'annual_volatility',
                  'sharpe_ratio', 'max_drawdown', 'win_rate', 'n_trades',
                  'trade_ratio']}
    if not per_stock_returns:
        return empty_ret

    # ---- 等权组合：按决策序号对齐 ----
    max_len = max(len(r) for r in per_stock_returns.values())
    portfolio_returns = []
    for k in range(max_len):
        day_r = [r[k] for r in per_stock_returns.values() if k < len(r)]
        if day_r:
            portfolio_returns.append(np.mean(day_r))
    portfolio_returns = np.array(portfolio_returns)

    if len(portfolio_returns) == 0:
        return empty_ret

    # ---- 组合权益曲线 ----
    portfolio_equity = np.cumprod(1 + portfolio_returns)
    cumulative_return = float(portfolio_equity[-1] - 1)

    # ---- 年化 ----
    # ★ 关键修改：用 decision_interval，不是 pred_horizon
    periods_per_year = trading_days / decision_interval
    n_periods = len(portfolio_returns)
    mean_period_return = float(np.mean(portfolio_returns))

    annual_return = float((1 + mean_period_return) ** periods_per_year - 1)
    annual_volatility = float(np.std(portfolio_returns) * np.sqrt(periods_per_year))

    if annual_volatility > 1e-8:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe_ratio = 0.0

    peak = np.maximum.accumulate(portfolio_equity)
    drawdown = (portfolio_equity - peak) / peak
    max_drawdown = float(drawdown.min())

    # ---- 统计：基于 signal != 0 ----
    all_active_returns = []
    n_trades = 0
    total_samples = 0
    total_signals = 0

    for sid in np.unique(stock_ids):
        mask = stock_ids == sid
        p = preds[mask]
        t = trues[mask]
        total_samples += len(p)

        signals = np.zeros_like(p)
        signals[p > signal_threshold] = 1.0
        signals[p < -signal_threshold] = -1.0
        active = signals != 0
        sig_count = int(active.sum())
        n_trades += sig_count
        total_signals += sig_count

        if sig_count > 0:
            sr = signals[active] * t[active] - cost_per_trade * np.abs(signals[active])
            sr = sr[np.isfinite(sr)]
            sr = np.clip(sr, -0.5, 0.5)
            all_active_returns.append(sr)

    trade_ratio = total_signals / total_samples if total_samples > 0 else 0.0

    if all_active_returns:
        all_active_returns = np.concatenate(all_active_returns)
        win_rate = float((all_active_returns > 0).mean())
    else:
        win_rate = 0.0

    return {
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'trade_ratio': float(trade_ratio),
    }