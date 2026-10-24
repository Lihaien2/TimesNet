import numpy as np

# General regression metrics (compatible with the original project API).
def RSE(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    return np.sqrt(np.sum((true - pred) ** 2)) / denom if denom > 0 else 0.0

def CORR(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt((((true - true.mean(0)) ** 2) * ((pred - pred.mean(0)) ** 2)).sum(0))
    with np.errstate(divide='ignore', invalid='ignore'):
        v = u / d
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0

def MAE(pred, true):
    return float(np.mean(np.abs(np.asarray(true) - np.asarray(pred))))

def MSE(pred, true):
    return float(np.mean((np.asarray(true) - np.asarray(pred)) ** 2))

def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))

def MAPE(pred, true, eps=1e-3):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.mean(np.abs((true - pred) / (np.abs(true) + eps))))

def MSPE(pred, true, eps=1e-3):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.mean(np.square((true - pred) / (np.abs(true) + eps))))

def metric(pred, true):
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true), MAPE(pred, true), MSPE(pred, true)

def compute_direction_accuracy_grouped(preds, trues, stock_ids, neutral_threshold=0.002):
    preds, trues, stock_ids = np.asarray(preds).reshape(-1), np.asarray(trues).reshape(-1), np.asarray(stock_ids).reshape(-1)
    if not (len(preds) == len(trues) == len(stock_ids)):
        raise ValueError('preds/trues/stock_ids 长度必须一致')
    active = np.isfinite(preds) & np.isfinite(trues) & (np.abs(trues) > neutral_threshold)
    return float(np.mean(np.sign(preds[active]) == np.sign(trues[active]))) if active.any() else 0.0

def backtest_strategy_grouped(preds, trues, stock_ids, signal_threshold=0.005,
                              cost_per_trade=0.0025, risk_free_rate=0.0,
                              trading_days=252, pred_horizon=10,
                              decision_interval=10, dates=None):
    """Date-aligned equal-weight portfolio backtest; no positional stock alignment/clipping."""
    if decision_interval < pred_horizon:
        raise ValueError(f'decision_interval ({decision_interval}) must be >= pred_horizon ({pred_horizon})')
    preds, trues, stock_ids = np.asarray(preds, float).reshape(-1), np.asarray(trues, float).reshape(-1), np.asarray(stock_ids).reshape(-1)
    if not (len(preds) == len(trues) == len(stock_ids)):
        raise ValueError('preds/trues/stock_ids 长度必须一致')
    if dates is None:
        raise ValueError('回测必须传入真实信号日期 dates；禁止按股票数组位置对齐')
    dates = np.asarray(dates).reshape(-1)
    if len(dates) != len(preds):
        raise ValueError('dates 长度必须与 preds/trues/stock_ids 一致')
    valid_date = np.asarray([x is not None and str(x) not in ('NaT', 'nan', 'None', '') for x in dates])
    valid = np.isfinite(preds) & np.isfinite(trues) & valid_date
    preds, trues, stock_ids, dates = preds[valid], trues[valid], stock_ids[valid], dates[valid]
    if len(preds) == 0:
        return _empty_metrics()
    signals = np.where(preds > signal_threshold, 1.0, np.where(preds < -signal_threshold, -1.0, 0.0))
    net = signals * trues - cost_per_trade * np.abs(signals)
    by_date = []
    for date in np.unique(dates):
        mask = dates == date
        r = net[mask]
        r = r[np.isfinite(r)]
        if len(r):
            by_date.append((date, float(np.mean(r))))
    by_date.sort(key=lambda x: x[0])
    if not by_date:
        return _empty_metrics()
    portfolio = np.asarray([x[1] for x in by_date], dtype=float)
    equity = np.cumprod(1.0 + portfolio)
    cumulative_return = float(equity[-1] - 1.0)
    periods_per_year = trading_days / float(decision_interval)
    mean_r = float(np.mean(portfolio))
    # Geometric annualization from the realized equity curve is consistent with
    # cumulative_return; it can still be unstable on a very short test sample.
    annual_return = float(equity[-1] ** (periods_per_year / len(portfolio)) - 1.0) if equity[-1] >= 0 else -1.0
    std = float(np.std(portfolio, ddof=1)) if len(portfolio) > 1 else 0.0
    annual_volatility = float(std * np.sqrt(periods_per_year))
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if risk_free_rate > -1 else risk_free_rate
    sharpe = float((mean_r - rf_period) / std * np.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    equity_with_initial = np.concatenate(([1.0], equity))
    drawdown = equity_with_initial / np.maximum.accumulate(equity_with_initial) - 1.0
    active = signals != 0
    active_returns = net[active]
    return {
        'cumulative_return': cumulative_return, 'annual_return': annual_return,
        'annual_volatility': annual_volatility, 'sharpe_ratio': sharpe,
        'max_drawdown': float(drawdown.min()),
        'win_rate': float(np.mean(active_returns > 0)) if len(active_returns) else 0.0,
        'n_trades': int(active.sum()), 'trade_ratio': float(active.mean()) if len(active) else 0.0,
        'n_portfolio_periods': int(len(portfolio)), 'n_signal_dates': int(len(by_date)),
    }

def _empty_metrics():
    return {k: 0.0 for k in ('cumulative_return', 'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'n_trades', 'trade_ratio', 'n_portfolio_periods', 'n_signal_dates')}
