"""
投资组合监控与再平衡引擎 (portfolio_rebalancer)
================================================

核心逻辑（参考富达/先锋等智能投顾标准再平衡方案）：

    1. 组合快照：当前权重 w_t = 当前市值 / 总净值
    2. 偏离检测：
         - 单资产阈值：|w_i - w^*_i| ≥ threshold_individual (默认 5%)
         - 整体欧式距离：sqrt(Σ(w_i - w^*_i)²) ≥ threshold_portfolio (默认 10%)
         - 或周期触发：距离上次 rebalance ≥ schedule_days
    3. 再平衡模式：
         - threshold       阈值触发
         - schedule        定期（月/季/半年）
         - threshold+schedule 混合
         - driftsync       漂移同向的资产同步调（避免反复交易）
    4. 交易成本约束：预估佣金 + 滑点 ≤ rebalancing_gain（从偏离估计），
       否则不触发，避免无意义换手。
    5. Rebalance → 订单列表：生成 buy/ sell 金额，用于 Freqtrade 机器人执行

典型用法::

    pr = PortfolioRebalancer(
        threshold_individual=0.05,
        threshold_portfolio=0.10,
        schedule_days=30,
        trading_cost_bps=15,           # 15 bps = 0.15%
        mode="threshold+schedule",
    )
    orders, info = pr.run(
        current_weights={"BTC":0.62,"ETH":0.31,"USDT":0.07},
        target_weights={"BTC":0.50,"ETH":0.30,"USDT":0.20},
        current_prices={"BTC":68000,"ETH":3500,"USDT":1.0},
        total_portfolio_value_usdt=100_000,
        last_rebalance_ts=ts_month_ago,
    )
    # orders → 可直接给 Freqtrade 下单
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 订单结构
# ---------------------------------------------------------------------------


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class RebalanceOrder:
    symbol: str
    side: OrderSide
    target_weight: float               # 0..1
    current_weight: float
    weight_delta: float                # 负=卖出，正=买入
    notional_usdt: float               # 名义金额（正=买，负=卖）
    base_qty: float = 0.0              # 对应数量（如 BTC 数量）= notional / price
    reason: str = ""                   # 触发原因


@dataclass
class RebalanceInfo:
    mode: str
    triggered: bool
    trigger_reasons: list[str]
    individual_max_deviation: float
    portfolio_deviation: float
    days_since_last_rebalance: int
    estimated_trading_cost_usdt: float
    estimated_gain_from_rebalance: float  # 启发式估计
    prior_weights: dict[str, float] = field(default_factory=dict)
    post_weights: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


RebalanceMode = Literal["threshold", "schedule", "threshold+schedule", "driftsync"]


class PortfolioRebalancer:
    """
    标准智能投顾式再平衡。
    """

    def __init__(
        self,
        threshold_individual: float = 0.05,
        threshold_portfolio: float = 0.10,
        schedule_days: int = 30,
        trading_cost_bps: float = 15.0,
        mode: RebalanceMode = "threshold+schedule",
        min_notional_usdt: float = 20.0,
        cash_buffer_weight: float = 0.0,
    ) -> None:
        if not (0 <= threshold_individual <= 1):
            raise ValueError("threshold_individual must lie in [0, 1]")
        if not (0 <= threshold_portfolio <= np.sqrt(2)):
            raise ValueError("threshold_portfolio out of range")
        if schedule_days <= 0:
            raise ValueError("schedule_days must be positive")
        self.t_ind = threshold_individual
        self.t_port = threshold_portfolio
        self.sched_days = schedule_days
        self.cost_bps = trading_cost_bps
        self.mode = mode
        self.min_notional = min_notional_usdt
        self.cash_buf = cash_buffer_weight

    # ---- 主入口 -------------------------------------------------------

    def run(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        total_portfolio_value_usdt: float,
        last_rebalance_ts: float | None = None,
        current_ts: float | None = None,
        allow_cash_adjustment: bool = True,
    ) -> tuple[list[RebalanceOrder], RebalanceInfo]:
        if total_portfolio_value_usdt <= 0:
            raise ValueError("portfolio value must be > 0")
        ts = current_ts or time.time()
        days_since = self._days_since(last_rebalance_ts, ts)

        # 对齐符号集合
        syms = list(set(current_weights) | set(target_weights))
        w_now = np.array([current_weights.get(s, 0.0) for s in syms], dtype=float)
        w_tgt = np.array([target_weights.get(s, 0.0) for s in syms], dtype=float)
        # 现金缓冲：从目标里抽出 cash_buf 比例给现金（或已包含）
        if self.cash_buf > 0 and allow_cash_adjustment:
            w_tgt = w_tgt * (1 - self.cash_buf)
            # 在 syms 中是否已有现金代号
            for cash_name in ("USDT", "USDC", "BUSD", "FIAT", "CASH"):
                if cash_name in syms:
                    idx = syms.index(cash_name)
                    w_tgt[idx] += self.cash_buf
                    break
        # 归一
        if w_now.sum() > 0:
            w_now = w_now / w_now.sum()
        if w_tgt.sum() > 0:
            w_tgt = w_tgt / w_tgt.sum()

        delta = w_tgt - w_now
        indiv_dev = float(np.max(np.abs(delta)))
        port_dev = float(np.sqrt(np.sum(delta ** 2)))

        # ---- 触发判断 ---------------------------------------------------
        reasons: list[str] = []
        trig_by_mode = False
        if self.mode in ("threshold", "threshold+schedule"):
            if indiv_dev >= self.t_ind:
                trig_by_mode = True
                reasons.append(f"单资产最大偏离 {indiv_dev:.2%} ≥ 阈值 {self.t_ind:.0%}")
            if port_dev >= self.t_port:
                trig_by_mode = True
                reasons.append(f"组合整体欧式偏离 {port_dev:.2%} ≥ 阈值 {self.t_port:.0%}")
        if self.mode in ("schedule", "threshold+schedule"):
            if days_since >= self.sched_days:
                trig_by_mode = True
                reasons.append(f"距离上次再平衡 {days_since} 天 ≥ 定期 {self.sched_days} 天")
        if self.mode == "driftsync":
            # 同向漂移指标：sign 相同的偏离加总
            pos = float(np.sum(np.maximum(delta, 0.0)))
            neg = float(-np.sum(np.minimum(delta, 0.0)))
            if min(pos, neg) >= max(self.t_ind, 0.02):
                trig_by_mode = True
                reasons.append(f"漂移同步检查：正总偏离 {pos:.2%} / 负总偏离 {neg:.2%}")

        # ---- 收益 / 成本权衡 ----
        # 预计换手 = 0.5 * Σ|Δ| × 总资产
        turnover_ratio = 0.5 * float(np.sum(np.abs(delta)))  # 一次买卖
        est_cost = turnover_ratio * total_portfolio_value_usdt * (self.cost_bps / 10_000)
        # 启发式收益：假设「偏离 × 历史 Sharpe」，这里用偏离作为效用提升 proxy
        est_gain = port_dev * total_portfolio_value_usdt * 0.25  # 相当于「25% σ 等价」的经验值
        if trig_by_mode and est_cost > est_gain * 0.8 and days_since < self.sched_days:
            reasons.append(f"但预计交易成本 {est_cost:.2f} ≈ 再平衡收益 {est_gain:.2f}，推迟触发")
            trig_by_mode = False

        orders: list[RebalanceOrder] = []
        post = dict(zip(syms, w_now))
        if trig_by_mode:
            for i, s in enumerate(syms):
                d_i = float(delta[i])
                notional = d_i * total_portfolio_value_usdt
                if abs(notional) < self.min_notional:
                    continue  # 金额过小忽略
                side = OrderSide.BUY if d_i > 0 else OrderSide.SELL
                price = float(current_prices.get(s, 0.0))
                qty = notional / price if price > 0 else 0.0
                orders.append(RebalanceOrder(
                    symbol=s, side=side,
                    target_weight=float(w_tgt[i]),
                    current_weight=float(w_now[i]),
                    weight_delta=d_i,
                    notional_usdt=notional, base_qty=qty,
                    reason="rebalance_to_target",
                ))
            post = dict(zip(syms, w_tgt))

        info = RebalanceInfo(
            mode=self.mode,
            triggered=trig_by_mode,
            trigger_reasons=reasons,
            individual_max_deviation=indiv_dev,
            portfolio_deviation=port_dev,
            days_since_last_rebalance=days_since,
            estimated_trading_cost_usdt=round(est_cost, 4),
            estimated_gain_from_rebalance=round(est_gain, 4),
            prior_weights=dict(zip(syms, w_now)),
            post_weights=post,
        )
        # 排序：先卖后买（避免资金不足）
        orders.sort(key=lambda o: (0 if o.side == OrderSide.SELL else 1, -abs(o.notional_usdt)))
        return orders, info

    # ---- 批量/回测再平衡 ---------------------------------------------

    def backtest_rebalance(
        self,
        weights_history: pd.DataFrame,      # index=日期，cols=symbols
        prices: pd.DataFrame,               # index=日期，cols=symbols
        initial_value: float = 100_000.0,
    ) -> tuple[pd.Series, list[RebalanceInfo]]:
        """
        将目标权重时序在价格历史上做「真实可执行」的再平衡回测：
        返回净值曲线 + 每个日期的再平衡决策信息。
        """
        import pandas as pd
        idx = weights_history.index
        # 保证 prices 长度/对齐一致（先按交集重排，缺失则 ffill/bfill）
        if not len(prices.index) == len(idx) or not (prices.index == idx).all():
            prices = prices.reindex(idx).ffill().bfill()
        value = initial_value
        w = np.zeros(weights_history.shape[1])  # 当前权重（起点全现金）
        vals = [value]
        infos: list[RebalanceInfo] = []
        last_ts: float | None = None
        for i, d in enumerate(idx):
            # 当日收盘后，根据价格变动更新权重（用当日价格回报）
            if i > 0:
                ret_i = prices.iloc[i].to_numpy(dtype=float) / np.clip(prices.iloc[i - 1].to_numpy(dtype=float), 1e-9, None) - 1
                pos_vals = value * w * (1 + ret_i)
                value = float(pos_vals.sum())
                w = np.where(value > 0, pos_vals / value, 0.0)
            target_w = weights_history.iloc[i].to_numpy(dtype=float)
            target_w = target_w / target_w.sum() if target_w.sum() > 0 else target_w
            # 将当前 weight dict 化
            cw = dict(zip(weights_history.columns, w))
            tw = dict(zip(weights_history.columns, target_w))
            pxs = dict(zip(prices.columns, prices.iloc[i].to_numpy(dtype=float)))
            ts = d.timestamp() if hasattr(d, "timestamp") else time.mktime(d.timetuple()) if not isinstance(d, (int, float)) else float(d)
            ords, info = self.run(
                cw, tw, pxs, value, last_rebalance_ts=last_ts, current_ts=ts,
            )
            infos.append(info)
            if info.triggered:
                w = np.array([info.post_weights.get(s, 0.0) for s in weights_history.columns], dtype=float)
                last_ts = ts
            vals.append(value)
        # 净值序列对齐 index + 1 个补充位
        eq_idx = list(idx) + [idx[-1] + pd.Timedelta(days=1)]
        series = pd.Series(vals, index=eq_idx[:len(vals)])
        return series, infos

    # ---- 辅助 --------------------------------------------------------

    @staticmethod
    def _days_since(ts_last: float | None, ts_now: float) -> int:
        if ts_last is None:
            return 10 ** 9  # 视为远大于 schedule
        return int(max(0, (ts_now - ts_last) // 86400))
