import numpy as np


# General regression metrics (compatible with the original project API).
def RSE(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    return (
        np.sqrt(np.sum((true - pred) ** 2)) / denom
        if denom > 0 else 0.0
    )


def CORR(pred, true):
    """Mean Pearson correlation across the final (feature) axis."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    if pred.shape != true.shape:
        raise ValueError(
            f'pred and true must have the same shape; '
            f'got {pred.shape} and {true.shape}'
        )

    pred_centered = pred - pred.mean(axis=0)
    true_centered = true - true.mean(axis=0)

    numerator = np.sum(true_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(true_centered ** 2, axis=0)
        * np.sum(pred_centered ** 2, axis=0)
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        values = numerator / denominator

    finite = np.isfinite(values)
    return float(np.mean(values[finite])) if finite.any() else 0.0


def MAE(pred, true):
    return float(
        np.mean(np.abs(np.asarray(true) - np.asarray(pred)))
    )


def MSE(pred, true):
    return float(
        np.mean((np.asarray(true) - np.asarray(pred)) ** 2)
    )


def RMSE(pred, true):
    return float(np.sqrt(MSE(pred, true)))


def MAPE(pred, true, eps=1e-3):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(
        np.mean(np.abs((true - pred) / (np.abs(true) + eps)))
    )


def MSPE(pred, true, eps=1e-3):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(
        np.mean(np.square((true - pred) / (np.abs(true) + eps)))
    )


def metric(pred, true):
    return (
        MAE(pred, true),
        MSE(pred, true),
        RMSE(pred, true),
        MAPE(pred, true),
        MSPE(pred, true),
    )


def compute_direction_accuracy_grouped(
    preds,
    trues,
    stock_ids,
    neutral_threshold=0.002,
):
    preds = np.asarray(preds, dtype=float).reshape(-1)
    trues = np.asarray(trues, dtype=float).reshape(-1)
    stock_ids = np.asarray(stock_ids).reshape(-1)

    if not (len(preds) == len(trues) == len(stock_ids)):
        raise ValueError(
            'preds/trues/stock_ids 长度必须一致'
        )

    if (
        not np.isfinite(neutral_threshold)
        or neutral_threshold < 0
    ):
        raise ValueError(
            'neutral_threshold 必须是非负有限数值'
        )

    active = (
        np.isfinite(preds)
        & np.isfinite(trues)
        & (np.abs(trues) > neutral_threshold)
    )

    return (
        float(
            np.mean(
                np.sign(preds[active])
                == np.sign(trues[active])
            )
        )
        if active.any()
        else 0.0
    )


def backtest_strategy_grouped(
    preds,
    trues,
    stock_ids,
    signal_threshold=0.005,
    cost_per_trade=0.0025,
    risk_free_rate=0.0,
    trading_days=252,
    pred_horizon=10,
    decision_interval=10,
    dates=None,
):
    """
    Backtest fixed-slot equal-weight positions grouped by actual signal date.

    Each eligible stock on a signal date represents one equal-weight
    portfolio slot. A +1/-1 signal takes a long/short position in that
    slot; a zero signal leaves that slot in cash.

    Therefore, trade_ratio also approximates the fraction of slots
    exposed to the market. Short P&L is modeled as -stock_return,
    before costs.
    """

    if (
        not isinstance(pred_horizon, (int, np.integer))
        or pred_horizon <= 0
    ):
        raise ValueError('pred_horizon 必须是正整数')

    if (
        not isinstance(decision_interval, (int, np.integer))
        or decision_interval <= 0
    ):
        raise ValueError('decision_interval 必须是正整数')

    if decision_interval < pred_horizon:
        raise ValueError(
            f'decision_interval ({decision_interval}) must be '
            f'>= pred_horizon ({pred_horizon})'
        )

    if (
        not isinstance(
            trading_days,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(trading_days)
        or trading_days <= 0
    ):
        raise ValueError(
            'trading_days 必须是正的有限数值'
        )

    if (
        not np.isfinite(signal_threshold)
        or signal_threshold < 0
    ):
        raise ValueError(
            'signal_threshold 必须是非负有限数值'
        )

    if (
        not np.isfinite(cost_per_trade)
        or cost_per_trade < 0
    ):
        raise ValueError(
            'cost_per_trade 必须是非负有限数值'
        )

    if (
        not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError(
            'risk_free_rate 必须是大于 -1 的有限年化利率'
        )

    preds = np.asarray(preds, dtype=float).reshape(-1)
    trues = np.asarray(trues, dtype=float).reshape(-1)
    stock_ids = np.asarray(stock_ids).reshape(-1)

    if not (len(preds) == len(trues) == len(stock_ids)):
        raise ValueError(
            'preds/trues/stock_ids 长度必须一致'
        )

    if dates is None:
        raise ValueError(
            '回测必须传入真实信号日期 dates；'
            '禁止按股票数组位置对齐'
        )

    dates = np.asarray(dates, dtype=object).reshape(-1)

    if len(dates) != len(preds):
        raise ValueError(
            'dates 长度必须与 preds/trues/stock_ids 一致'
        )

    # Normalize dates to YYYY-MM-DD so equivalent timestamps
    # cannot become separate portfolio periods.
    date_keys = np.empty(len(dates), dtype=object)

    for i, value in enumerate(dates):
        if (
            value is None
            or str(value).strip().lower()
            in ('', 'nat', 'nan', 'none')
        ):
            date_keys[i] = None
            continue

        try:
            day = np.datetime64(str(value)[:10], 'D')
            date_keys[i] = (
                None if np.isnat(day) else str(day)
            )
        except (TypeError, ValueError):
            date_keys[i] = None

    valid_observation = (
        np.isfinite(preds)
        & np.isfinite(trues)
        & np.asarray(
            [d is not None for d in date_keys],
            dtype=bool,
        )
    )

    # A valid return without stock identity cannot be audited as
    # an equal-weight stock slot.
    id_missing = np.asarray(
        [
            sid is None
            or str(sid).strip().lower()
            in ('', 'nan', 'none')
            for sid in stock_ids
        ],
        dtype=bool,
    )

    if np.any(valid_observation & id_missing):
        raise ValueError(
            '有效回测样本缺少 stock_id，'
            '无法确认等权组合权重'
        )

    valid = valid_observation

    preds = preds[valid]
    trues = trues[valid]
    stock_ids = stock_ids[valid]
    date_keys = date_keys[valid]

    if len(preds) == 0:
        return _empty_metrics()

    # Each (signal date, stock) must contribute no more than
    # one observation to the date's equal-weight portfolio.
    seen = set()

    for date, sid in zip(date_keys, stock_ids):
        key = (date, str(sid))

        if key in seen:
            raise ValueError(
                f'发现重复的 (signal_date, stock_id) 样本：'
                f'date={date}, stock_id={sid!r}；'
                '重复记录会改变等权组合权重。'
            )

        seen.add(key)

    signals = np.where(
        preds > signal_threshold,
        1.0,
        np.where(
            preds < -signal_threshold,
            -1.0,
            0.0,
        ),
    )

    # Charge costs only to active stock slots.
    # Date-level averaging spreads the total portfolio return
    # across all eligible slots, including slots held in cash.
    net = (
        signals * trues
        - cost_per_trade * np.abs(signals)
    )

    by_date = []

    for date in sorted(set(date_keys)):
        mask = date_keys == date
        r = net[mask]
        r = r[np.isfinite(r)]

        if len(r):
            by_date.append((date, float(np.mean(r))))

    if not by_date:
        return _empty_metrics()

    portfolio = np.asarray(
        [value for _, value in by_date],
        dtype=float,
    )

    if np.any(portfolio <= -1.0):
        raise ValueError(
            '策略组合存在单周期收益 <= -100%，'
            '净值无法按普通复利定义'
        )

    equity = np.cumprod(1.0 + portfolio)
    cumulative_return = float(equity[-1] - 1.0)

    periods_per_year = (
        float(trading_days) / float(decision_interval)
    )

    mean_r = float(np.mean(portfolio))

    annual_return = float(
        equity[-1] ** (
            periods_per_year / len(portfolio)
        ) - 1.0
    )

    std = (
        float(np.std(portfolio, ddof=1))
        if len(portfolio) > 1
        else 0.0
    )

    annual_volatility = float(
        std * np.sqrt(periods_per_year)
    )

    rf_period = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    sharpe = (
        float(
            (mean_r - rf_period)
            / std
            * np.sqrt(periods_per_year)
        )
        if std > 1e-12
        else 0.0
    )

    equity_with_initial = np.concatenate(
        ([1.0], equity)
    )

    drawdown = (
        equity_with_initial
        / np.maximum.accumulate(equity_with_initial)
        - 1.0
    )

    active = signals != 0
    active_returns = net[active]

    return {
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': float(drawdown.min()),
        'win_rate': (
            float(np.mean(active_returns > 0))
            if len(active_returns)
            else 0.0
        ),
        'n_trades': int(active.sum()),
        'trade_ratio': (
            float(active.mean())
            if len(active)
            else 0.0
        ),
        'n_portfolio_periods': int(len(portfolio)),
        'n_signal_dates': int(len(by_date)),
    }


def _empty_metrics():
    return {
        k: 0.0
        for k in (
            'cumulative_return',
            'annual_return',
            'annual_volatility',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'n_trades',
            'trade_ratio',
            'n_portfolio_periods',
            'n_signal_dates',
        )
    }