"""
绩效报告与收益归因引擎 (report_engine)
========================================

提供组合全周期的绩效报告与收益归因（Brinson + 风险分解）：

    ReportEngine.summarize(equity_curve, benchmark=None)
        → 计算：累计收益、年化、波动、夏普、索提诺、卡玛、最大回撤（起止日期、时长）、
              VaR（参数 / 历史 / CVaR）、胜率、盈亏比、月度热力矩阵

    ReportEngine.brinson(portfolio_weights, benchmark_weights, returns_by_period)
        → 每期 Brinson 分解 + 累计（配置/选股/交互效应）

    ReportEngine.risk_decomposition(weights, factor_cov, factor_names)
        → 风险按因子维度分解（欧拉分解：风险贡献比例）

输出为 JSON + DataFrame 混合，便于：
    - API 直接吐给前端
    - 转 Markdown / HTML
    - 直接接 LLM 生成自然语言解读
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 绩效输出结构
# ---------------------------------------------------------------------------


@dataclass
class PerformanceSummary:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_info: dict[str, Any]        # start_peak / end_trough / recovery / duration_days
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float
    monthly_returns: pd.DataFrame           # 行=年，列=月
    drawdown_series: pd.Series              # 回撤序列（0.0 到 -0.xx）
    equity_series: pd.Series
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrinsonResult:
    period_breakdown: pd.DataFrame          # index=period, cols=allocation/selection/interaction/active
    cumulative: pd.Series                   # 累计三条线
    total_active: float


@dataclass
class RiskDecomposition:
    total_volatility: float
    factor_contributions: dict[str, float]  # 各因子绝对风险贡献（RC_i）
    factor_contrib_pct: dict[str, float]    # RC_i / σ²
    per_asset_contributions: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class ReportEngine:
    """绩效 & 归因综合引擎。"""

    def __init__(self, periods_per_year: float = 365.0, risk_free_rate: float = 0.04) -> None:
        self.ppy = periods_per_year
        self.rf = risk_free_rate

    # ---- 1) 绩效总览 -------------------------------------------------

    def summarize(
        self,
        equity: pd.Series,
        benchmark: pd.Series | None = None,
    ) -> PerformanceSummary:
        eq = equity.dropna().sort_index().astype(float)
        if eq.empty:
            raise ValueError("Empty equity series")
        if eq.iloc[0] <= 0:
            raise ValueError("Equity must start positive (e.g., 1.0 or 100_000)")
        total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        # 按时间频率估计 periods_per_year（自动校正）
        est_ppy = self._estimate_periods_per_year(eq)
        n_periods = len(eq) - 1
        ann = (1 + total_ret) ** (est_ppy / max(n_periods, 1)) - 1
        # 区间收益
        rets = eq.pct_change().dropna()
        rets_excess = rets - (self.rf / est_ppy)
        vol = float(rets.std(ddof=0)) * np.sqrt(est_ppy)
        sharpe = float((rets_excess.mean() * est_ppy) / (rets.std(ddof=0) * np.sqrt(est_ppy))) if vol > 0 else 0.0
        # Sortino
        down = rets[rets < 0]
        down_vol = float(down.std(ddof=0) * np.sqrt(est_ppy)) if len(down) > 1 else float("nan")
        sortino = float((rets.mean() - self.rf / est_ppy) * est_ppy / down_vol) if down_vol and down_vol > 0 else float("nan")
        # 最大回撤
        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        max_dd = float(dd.min())
        mdd_info = self._mdd_details(eq)
        calmar = float(ann / abs(max_dd)) if abs(max_dd) > 1e-9 else float("inf")
        # VaR / CVaR（95%，历史模拟）
        var95 = float(np.quantile(rets, 0.05))
        cvar95 = float(rets[rets <= var95].mean()) if np.any(rets <= var95) else var95
        # 胜率 / 盈亏比
        wins = (rets > 0).sum()
        total = (rets != 0).sum()
        win_rate = float(wins / total) if total > 0 else 0.0
        avg_win = float(rets[rets > 0].mean()) if wins > 0 else 0.0
        avg_loss = float(abs(rets[rets < 0].mean())) if (rets < 0).any() else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")
        # 月度收益矩阵（年×月）
        month_rets = self._monthly_returns(eq)
        meta: dict[str, Any] = {}
        if benchmark is not None:
            bm = benchmark.dropna().sort_index().reindex(eq.index).ffill().bfill()
            if not bm.empty and bm.iloc[0] > 0:
                bm_ret = float(bm.iloc[-1] / bm.iloc[0] - 1.0)
                meta["benchmark_total_return"] = bm_ret
                # 信息比率（跟踪误差）
                diff = eq.pct_change().dropna() - bm.pct_change().dropna().reindex(rets.index).fillna(0.0)
                te = float(diff.std(ddof=0) * np.sqrt(est_ppy))
                active_ann = float((1 + (eq.iloc[-1] / eq.iloc[0] - 1 - bm_ret)) ** (est_ppy / max(n_periods, 1)) - 1)
                meta["information_ratio"] = active_ann / te if te > 0 else float("nan")
                meta["beta"] = float((rets.cov(bm.pct_change().dropna().reindex(rets.index).fillna(0.0)))) / float(bm.pct_change().var(ddof=0)) if bm.pct_change().var(ddof=0) > 0 else 0.0

        return PerformanceSummary(
            total_return=total_ret, annual_return=float(ann), annual_volatility=vol,
            sharpe_ratio=sharpe, sortino_ratio=sortino, calmar_ratio=calmar,
            max_drawdown=max_dd, max_drawdown_info=mdd_info,
            var_95=var95, cvar_95=cvar95, win_rate=win_rate, profit_factor=profit_factor,
            monthly_returns=month_rets, drawdown_series=dd, equity_series=eq, meta=meta,
        )

    # ---- 2) Brinson 归因 ------------------------------------------------

    def brinson(
        self,
        weights_portfolio: pd.DataFrame,     # index=period, cols=asset (period-end 再平衡权重)
        weights_benchmark: pd.DataFrame,     # 同上
        asset_returns: pd.DataFrame,         # index=period, cols=asset (该 period 的总收益)
    ) -> BrinsonResult:
        periods = weights_portfolio.index
        syms = [c for c in weights_portfolio.columns if c in asset_returns.columns and c in weights_benchmark.columns]
        wp = weights_portfolio[syms].reindex(periods).fillna(0.0).to_numpy(dtype=float)
        wb = weights_benchmark[syms].reindex(periods).fillna(0.0).to_numpy(dtype=float)
        r = asset_returns[syms].reindex(periods).fillna(0.0).to_numpy(dtype=float)

        allocation = np.zeros(len(periods))
        selection = np.zeros(len(periods))
        interaction = np.zeros(len(periods))
        active = np.zeros(len(periods))
        for t in range(len(periods)):
            wp_t = wp[t]; wb_t = wb[t]; r_t = r[t]
            rb_total = float(wb_t @ r_t)
            # 按各资产展开
            alloc_t = (wp_t - wb_t) * (r_t - rb_total)      # 简化的 Brinson 配置（去掉「行业平均」概念 → 资产级直接展开）
            # selection：基准权重下，投资收益 - 基准收益（选股差异）
            sel_t = wb_t * 0.0  # 资产级：这里 rp=rb 即没有选股差异 → 为 0；保留变量结构
            inter_t = (wp_t - wb_t) * (r_t - rb_total)       # 交互：超额权重 × 资产超额相对
            # 令三项和等于 active: w_p'r - w_b'r
            act = float((wp_t - wb_t) @ r_t)
            selection[t] = act - (alloc_t.sum() + inter_t.sum())
            allocation[t] = float(alloc_t.sum())
            interaction[t] = float(inter_t.sum())
            active[t] = act
        df = pd.DataFrame(
            {"allocation": allocation, "selection": selection, "interaction": interaction, "active": active},
            index=periods,
        )
        cum = df.cumsum().iloc[-1] if len(df) else pd.Series(dtype=float)
        total_active = float(cum.get("active", 0.0))
        return BrinsonResult(period_breakdown=df, cumulative=cum, total_active=total_active)

    # ---- 3) 风险分解 --------------------------------------------------

    def risk_decomposition(
        self,
        weights: dict[str, float],
        factor_covariance: np.ndarray | pd.DataFrame,
        factor_names: list[str] | None = None,
        asset_names: list[str] | None = None,
        asset_covariance: np.ndarray | pd.DataFrame | None = None,
    ) -> RiskDecomposition:
        # 1) 因子维度：w 对因子 → 组合方差 = w'XFX'w；分解按因子
        if isinstance(factor_covariance, pd.DataFrame):
            if factor_names is None:
                factor_names = list(factor_covariance.columns)
            F = factor_covariance.to_numpy(dtype=float)
        else:
            F = np.asarray(factor_covariance, dtype=float)
        if factor_names is None:
            factor_names = [f"F{i+1}" for i in range(F.shape[0])]
        # weights_per_factor（假设传入的 weights 就是因子暴露权重：组合暴露向量 β = w）
        beta = np.array([weights.get(f, 0.0) for f in factor_names], dtype=float)
        port_var = float(beta @ F @ beta)
        sigma = float(np.sqrt(max(port_var, 0.0)))
        # 欧拉分解：RC_i = β_i · (Fβ)_i
        Fb = F @ beta
        rc_factor = beta * Fb
        contrib_abs = {f: float(rc_factor[i]) for i, f in enumerate(factor_names)}
        contrib_pct = {f: (float(v / port_var) if port_var > 0 else 0.0) for f, v in contrib_abs.items()}
        # 2) 资产级（可选）
        asset_rc: dict[str, float] | None = None
        if asset_covariance is not None and asset_names is not None:
            if isinstance(asset_covariance, pd.DataFrame):
                A = asset_covariance.to_numpy(dtype=float)
            else:
                A = np.asarray(asset_covariance, dtype=float)
            w = np.array([weights.get(a, 0.0) for a in asset_names], dtype=float)
            rc_asset = w * (A @ w)
            asset_rc = {a: float(rc_asset[i]) for i, a in enumerate(asset_names)}
            # 重新校准 σ（若给出资产协方差）
            sigma = float(np.sqrt(max(float(w @ A @ w), 0.0)))
        return RiskDecomposition(
            total_volatility=sigma,
            factor_contributions=contrib_abs,
            factor_contrib_pct=contrib_pct,
            per_asset_contributions=asset_rc,
        )

    # ---- 4) 一键生成 Markdown 报告（给 LLM 或前端用）---------------

    @staticmethod
    def to_markdown(summary: PerformanceSummary, title: str = "Portfolio Report") -> str:
        mdd = summary.max_drawdown_info
        lines = [f"# {title}", ""]
        lines.append("## 绩效概览")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        def _fmt(x, pct=False, decimals=2):
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "N/A"
            return f"{x:.{decimals}{'%' if pct else 'f'}}"
        lines.append(f"| 累计收益 | {_fmt(summary.total_return*100)}% |")
        lines.append(f"| 年化收益 | {_fmt(summary.annual_return*100)}% |")
        lines.append(f"| 年化波动率 | {_fmt(summary.annual_volatility*100)}% |")
        lines.append(f"| 夏普比率 | {_fmt(summary.sharpe_ratio)} |")
        lines.append(f"| 索提诺比率 | {_fmt(summary.sortino_ratio)} |")
        lines.append(f"| 卡玛比率 | {_fmt(summary.calmar_ratio)} |")
        lines.append(f"| 最大回撤 | {_fmt(summary.max_drawdown*100)}% |")
        lines.append(f"| VaR(95%) | {_fmt(summary.var_95*100)}% |")
        lines.append(f"| CVaR(95%) | {_fmt(summary.cvar_95*100)}% |")
        lines.append(f"| 胜率 | {_fmt(summary.win_rate*100)}% |")
        lines.append(f"| 盈亏比 | {_fmt(summary.profit_factor)} |")
        lines.append("")
        if mdd:
            lines.append("## 最大回撤明细")
            lines.append(f"- 峰值日期：{mdd.get('peak_date','')}")
            lines.append(f"- 谷值日期：{mdd.get('trough_date','')}")
            lines.append(f"- 修复日期：{mdd.get('recovery_date','(未修复)')}")
            lines.append(f"- 回撤持续天数：{mdd.get('duration_days','')} 天")
            lines.append("")
        if summary.meta:
            lines.append("## 基准对比")
            for k, v in summary.meta.items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")
        lines.append("## 月度收益矩阵（%）")
        lines.append("")
        m = summary.monthly_returns * 100
        try:
            lines.append(m.round(2).fillna("").to_markdown())
        except Exception:
            # tabulate 不可用时 fallback：手写 pipe 分隔表
            cols = list(m.columns)
            lines.append("| year | " + " | ".join(cols) + " |")
            lines.append("|------|" + "|".join(["---"] * len(cols)) + "|")
            for year, row in m.iterrows():
                vals = [("" if pd.isna(v) else f"{v:.2f}") for v in row.tolist()]
                lines.append(f"| {year} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    # ---- 内部工具 ----------------------------------------------------

    @staticmethod
    def _estimate_periods_per_year(eq: pd.Series) -> float:
        if len(eq) < 2:
            return 365.0
        idx = eq.index
        try:
            diffs = pd.Series(idx).diff().dropna().dt.days  # type: ignore[union-attr]
            median_day = float(np.clip(diffs.median(), 1/24, 365))
        except Exception:
            median_day = 1.0
        return 365.0 / median_day if median_day > 0 else 365.0

    @staticmethod
    def _mdd_details(eq: pd.Series) -> dict[str, Any]:
        roll_max = eq.cummax()
        dd = eq / roll_max - 1.0
        k = int(np.argmin(dd.to_numpy(dtype=float)))
        if k <= 0:
            return {}
        peak = int(np.argmax(eq.iloc[: k + 1].to_numpy(dtype=float)))
        # 修复：trough 之后首次超过峰值
        after = eq.iloc[k:]
        recovery_mask = after >= eq.iloc[peak]
        recovery_idx = int(np.argmax(recovery_mask.to_numpy())) if recovery_mask.any() else None
        peak_date = eq.index[peak]
        trough_date = eq.index[k]
        rec_date = after.index[recovery_idx] if recovery_idx is not None and recovery_idx > 0 else None
        duration_days = None
        try:
            duration_days = (trough_date - peak_date).days
        except Exception:
            duration_days = k - peak
        info = {
            "peak_date": peak_date,
            "trough_date": trough_date,
            "recovery_date": rec_date,
            "duration_days": duration_days,
        }
        return info

    @staticmethod
    def _monthly_returns(eq: pd.Series) -> pd.DataFrame:
        # 月末净值 → 月度收益 → 按年/月展开
        month_end = eq.resample("ME").last()
        mr = month_end.pct_change().dropna()
        rows = {}
        for ts, r in mr.items():
            yr = ts.year
            mo = ts.month
            rows.setdefault(yr, {})[mo] = float(r)
        df = pd.DataFrame.from_dict(rows, orient="index", columns=list(range(1, 13)))
        df.index.name = "year"
        df.columns = [f"M{c}" for c in df.columns]
        return df.sort_index()
