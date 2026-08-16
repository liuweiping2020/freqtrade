"""
因子分析与回测模块 (factor_analyzer)
=====================================

对标 Alphalens / AlphaPurify FactorAnalyzer 核心功能：

    1. IC / Rank IC 时序（截面皮尔逊 / 斯皮尔曼）+ 统计检验
    2. 分位数组合回测（Quantile Backtest）：按因子排序分 5/10 组
       → 每组净值曲线、年化收益、夏普、最大回撤、胜率
       → 多空组合（高 - 低）绩效
    3. Alpha 衰减（Decay）：因子对 t, t+1, …, t+k 日收益的预测力变化
    4. 因子稳定性：截面换手率（相邻日期因子秩相关系数、分位组留存率）
    5. 事件研究（Event Study）：调仓后持有 H 日的累计收益曲线

输入约定：
    factor_panel : pd.DataFrame  MultiIndex [date, asset] × 单因子列（或一列 Series）
    forward_returns : pd.DataFrame MultiIndex [date, asset] × N 列 holding period 收益
        常用列名示例：ret_1d / ret_5d / ret_10d / ret_20d
        含义：在 date 截面末买入，持有对应期数的**累计**收益（几何）

典型用法::

    from freqtrade.robo_advisor import FactorAnalyzer
    result = FactorAnalyzer(quantiles=5).run(factor_panel, forward_returns)
    result.ic_summary()          # IC 统计总览
    result.quantile_performance()# 分位数组合绩效
    result.decay()               # Alpha 衰减
    result.tear_sheet()          # 一键综合诊断

与 Freqtrade 对接：
    因子清洗好后 → 用 to_freqai_feature_dataframe 后作为 FreqAI 特征
    但本模块做「因子研究」，直接喂原始因子面板即可，无需走 FreqAI 重训练流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _ensure_series(df_or_s: pd.DataFrame | pd.Series, name: str = "factor") -> pd.Series:
    if isinstance(df_or_s, pd.Series):
        return df_or_s
    if isinstance(df_or_s, pd.DataFrame):
        if df_or_s.shape[1] == 1:
            return df_or_s.iloc[:, 0]
        # 若多列：取名为 name 的列
        if name in df_or_s.columns:
            return df_or_s[name]
        raise ValueError(f"DataFrame has {df_or_s.shape[1]} columns; specify which one via 'factor_name'")
    raise TypeError("factor_panel must be Series or DataFrame")


def _rank(x: pd.Series) -> pd.Series:
    return x.rank(method="average")


def _t_test_onesided(x: np.ndarray, null_mean: float = 0.0) -> tuple[float, float]:
    """单侧 t 检验（x!= null_mean），返回 (t_stat, p_value)。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return 0.0, 1.0
    m = float(x.mean() - null_mean)
    s = float(x.std(ddof=1))
    if s == 0:
        return 0.0, 1.0
    t = m / (s / np.sqrt(n))
    # 近似 p 值（用误差函数近似正态 cdf 对于 n>30 足够，n 小用 t 分布）
    try:
        from scipy import stats as _s  # lazy
        p = float(2 * _s.t.sf(abs(t), df=n - 1))
    except Exception:
        # fallback: 正态近似
        from math import erf, sqrt
        z = t
        p = float(2 * (0.5 * (1 + erf(abs(z) / sqrt(2)))) - 1)
        p = 1.0 - p
    return t, p


def _max_drawdown(equity: pd.Series) -> tuple[float, int, int]:
    """最大回撤比例 (0~-1) 及其起/止下标。"""
    eq = equity.to_numpy(dtype=float)
    roll_max = np.maximum.accumulate(eq)
    dd = eq / roll_max - 1.0
    k = int(np.argmin(dd))
    if k <= 0:
        return float(dd[k]), 0, k
    peak = int(np.argmax(eq[: k + 1]))
    return float(dd[k]), peak, k


def _annualize_return(total_ret: float, periods_per_year: float) -> float:
    # 几何年化；极端范围退化到简单年化避免数值溢出
    if total_ret <= -1:
        return float("nan")
    terminal = 1.0 + total_ret
    # 若 (1+R) 太大则用简单线性年化近似（log 也可）
    if terminal > 1e20:
        return float(np.log(terminal) * periods_per_year)
    try:
        return float(terminal ** periods_per_year - 1.0)
    except OverflowError:
        return float(np.log(terminal) * periods_per_year)


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------


@dataclass
class FactorAnalysisResult:
    factor_name: str
    periods_per_year: float
    ic_df: pd.DataFrame                             # columns: ic / rank_ic
    quantile_returns: dict[str, pd.DataFrame]        # key: holding_period, value: (dates × Q1..Qn + LS)
    quantile_turnover: pd.DataFrame                  # dates × Q1..Qn 组内留存率(%)
    decay_df: pd.DataFrame                           # index: lag → mean_ic / mean_rank_ic
    event_curve: pd.DataFrame                        # index: holding_days → avg_alpha(Q_high - Q_low)
    meta: dict[str, Any] = field(default_factory=dict)
    _quantiles: int = 5

    # --- 输出 API ---------------------------------------------------------

    def ic_summary(self) -> pd.Series:
        """IC / RankIC 统计（均值/IR/t/p/%>0/自相关）。"""
        ic = self.ic_df["ic"].dropna()
        ric = self.ic_df["rank_ic"].dropna()
        rows = {}
        for name, s in [("IC", ic), ("RankIC", ric)]:
            if s.empty:
                continue
            t, p = _t_test_onesided(s.to_numpy(), 0.0)
            ac1 = float(s.autocorr(lag=1)) if len(s) > 2 else float("nan")
            rows[f"{name}_mean"] = float(s.mean())
            rows[f"{name}_std"] = float(s.std(ddof=0))
            rows[f"{name}_IR"] = float(s.mean() / s.std(ddof=0)) if s.std(ddof=0) > 0 else 0.0
            rows[f"{name}_tstat"] = t
            rows[f"{name}_pval"] = p
            rows[f"{name}_pct_gt_0"] = float((s > 0).mean())
            rows[f"{name}_AC1"] = ac1
        return pd.Series(rows, name=self.factor_name).round(4)

    def quantile_performance(self, holding_period: str | None = None) -> pd.DataFrame:
        """分位数组合绩效表：年化收益/波动/夏普/最大回撤/胜率/Calmar。"""
        hp = holding_period or next(iter(self.quantile_returns))
        rets = self.quantile_returns[hp]
        rows = {}
        # 提取 horizon（估计的 periods_per_year 折算）：
        # 若持有期 H 天，则一年约 periods_per_year / H 个独立重平衡
        H_str = hp.replace("ret_", "").replace("d", "")
        try:
            H = int(H_str)
        except ValueError:
            H = 1
        freq_factor = self.periods_per_year / max(H, 1)
        for col in rets.columns:
            s = rets[col].dropna()
            if s.empty:
                continue
            eq = (1 + s).cumprod()
            total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
            ann = _annualize_return(total, freq_factor)
            vol = float(s.std(ddof=0) * np.sqrt(freq_factor))
            sharpe = ann / vol if vol > 0 else 0.0
            mdd, _, _ = _max_drawdown(eq)
            win_rate = float((s > 0).mean())
            calmar = ann / abs(mdd) if abs(mdd) > 1e-12 else float("inf")
            rows[col] = {
                "total_return": round(total, 4),
                "annual_return": round(ann, 4),
                "annual_vol": round(vol, 4),
                "sharpe": round(sharpe, 4),
                "max_drawdown": round(mdd, 4),
                "calmar": round(calmar, 4),
                "win_rate": round(win_rate, 4),
                "periods": int(s.shape[0]),
            }
        return pd.DataFrame(rows).T

    def decay(self) -> pd.DataFrame:
        return self.decay_df.round(4)

    def turnover_summary(self) -> pd.Series:
        """各组的平均留存率（1-该值=需要换出的比例）。"""
        return (self.quantile_turnover.mean() / 100.0).round(4)

    def tear_sheet(self) -> dict[str, Any]:
        """综合诊断（便于批量因子评估）。"""
        ic_sum = self.ic_summary()
        qp = self.quantile_performance()
        # Long-Short 分层单调性：Q1..Qn 斜率系数
        try:
            cols = [f"Q{i+1}" for i in range(self._quantiles)]
            ann_rets = qp.loc[cols, "annual_return"].to_numpy()
            xs = np.arange(1, len(cols) + 1, dtype=float)
            slope = float(np.polyfit(xs, ann_rets, 1)[0])
        except Exception:
            slope = float("nan")
        ls_key = "LS"
        ls_sharpe = float(qp.loc[ls_key, "sharpe"]) if ls_key in qp.index else float("nan")
        ls_dd = float(qp.loc[ls_key, "max_drawdown"]) if ls_key in qp.index else float("nan")
        return {
            "factor": self.factor_name,
            "RankIC_mean": ic_sum.get("RankIC_mean", np.nan),
            "RankIC_IR": ic_sum.get("RankIC_IR", np.nan),
            "RankIC_pval": ic_sum.get("RankIC_pval", np.nan),
            "LS_sharpe": ls_sharpe,
            "LS_max_drawdown": ls_dd,
            "quantile_slope": round(slope, 6),
            "quantile_performance": qp,
            "ic_summary": ic_sum,
            "decay": self.decay(),
        }


# ---------------------------------------------------------------------------
# FactorAnalyzer 主类
# ---------------------------------------------------------------------------


class FactorAnalyzer:
    """
    因子诊断主类。

    Parameters:
        quantiles       分位数，一般 5 或 10（默认 5）
        periods_per_year 年化换算，日频→252（A股）/ 365（crypto），小时频→365*24 …
        long_quantile   "top"(默认 QN 做多头) / "bottom"(反向因子 Q1 做多头)
    """

    def __init__(
        self,
        quantiles: int = 5,
        periods_per_year: float = 365.0,
        long_quantile: Literal["top", "bottom"] = "top",
        min_assets_per_date: int = 30,
    ) -> None:
        if quantiles < 2:
            raise ValueError("quantiles must be >= 2")
        self.quantiles = quantiles
        self.periods_per_year = periods_per_year
        self.long_quantile = long_quantile
        self.min_assets = min_assets_per_date

    # ------------------------------------------------------------------
    # 核心入口
    # ------------------------------------------------------------------

    def run(
        self,
        factor_panel: pd.DataFrame | pd.Series,
        forward_returns: pd.DataFrame,
        *,
        factor_name: str = "factor",
        lags: Sequence[int] = (0, 1, 2, 3, 5, 10, 20),
        event_horizon: int = 20,
    ) -> FactorAnalysisResult:
        factor = _ensure_series(factor_panel, factor_name).rename(factor_name)

        # 对齐 index
        common_idx = factor.index.intersection(forward_returns.index)
        if common_idx.empty:
            raise ValueError("factor_panel and forward_returns have no matching MultiIndex")
        f_aligned = factor.reindex(common_idx)
        fr_aligned = forward_returns.reindex(common_idx)

        # 1) IC 时序
        ic_df = self._compute_ic_series(f_aligned, fr_aligned)

        # 2) 分位数回测：每个 holding period 一列
        quantile_returns, q_turnover = self._quantile_backtest(f_aligned, fr_aligned)

        # 3) Alpha 衰减：对不同 lag 后的 factor → ret_1d 的 IC
        if "ret_1d" in fr_aligned.columns:
            decay_df = self._alpha_decay(f_aligned, fr_aligned["ret_1d"], lags=lags)
        else:
            # fallback: 用第一列收益
            decay_df = self._alpha_decay(f_aligned, fr_aligned.iloc[:, 0], lags=lags)

        # 4) Event Study：调仓后 1~event_horizon 天累计收益（按高-低分组）
        # 需要每日调仓（重抽样）实现：这里用多 horizon forward_returns 近似
        hp_cols = [c for c in fr_aligned.columns if c.startswith("ret_")]
        event_curve = self._event_study(f_aligned, fr_aligned, hp_cols, event_horizon)

        return FactorAnalysisResult(
            factor_name=factor_name,
            periods_per_year=self.periods_per_year,
            ic_df=ic_df,
            quantile_returns=quantile_returns,
            quantile_turnover=q_turnover,
            decay_df=decay_df,
            event_curve=event_curve,
            _quantiles=self.quantiles,
            meta={"long_quantile": self.long_quantile},
        )

    # ------------------------------------------------------------------
    # 子步骤实现
    # ------------------------------------------------------------------

    def _compute_ic_series(
        self, factor: pd.Series, fr: pd.DataFrame
    ) -> pd.DataFrame:
        """按日截面计算 IC(Pearson) 与 RankIC(Spearman) 与各持有期对齐后的均值。"""
        # 对每个持有期收益列分别计算，最终取平均作为每日单一 IC 值
        records: list[dict[str, Any]] = []
        for d, xs in factor.groupby(level=0, sort=False):
            xs = xs.dropna()
            if xs.shape[0] < self.min_assets:
                continue
            fr_d = fr.loc[d]
            # 找每个持有期的收益
            per_col_ic = []
            per_col_ric = []
            for c in fr.columns:
                if not c.startswith("ret_"):
                    continue
                y = fr_d[c] if isinstance(fr_d, pd.DataFrame) else fr_d
                y = y.reindex(xs.index).dropna()
                if y.shape[0] < self.min_assets:
                    continue
                fx = xs.reindex(y.index).astype(float)
                yy = y.astype(float)
                ic = float(fx.corr(yy, method="pearson"))
                ric = float(fx.corr(yy, method="spearman"))
                if not np.isnan(ic):
                    per_col_ic.append(ic)
                if not np.isnan(ric):
                    per_col_ric.append(ric)
            if not per_col_ic:
                continue
            records.append({
                "date": d,
                "ic": float(np.nanmean(per_col_ic)),
                "rank_ic": float(np.nanmean(per_col_ric)),
            })
        if not records:
            return pd.DataFrame(columns=["date", "ic", "rank_ic"]).set_index("date")
        df = pd.DataFrame(records).set_index("date").sort_index()
        return df

    def _quantile_backtest(
        self, factor: pd.Series, fr: pd.DataFrame
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """按分位数多空组合回测，返回 {holding_period: DataFrame(date × Q1..Qn LS)} 和 换手表。"""
        Q = self.quantiles
        hp_cols = [c for c in fr.columns if c.startswith("ret_")]
        rets_per_hp: dict[str, pd.DataFrame] = {}
        # 每期组别 DataFrame:  index=date × asset, value=group_label
        groups = []
        dates_order = []
        for d, xs in factor.groupby(level=0, sort=False):
            xs = xs.dropna()
            if xs.shape[0] < self.min_assets:
                continue
            dates_order.append(d)
            rk = _rank(xs)
            # 切分 Q 组：qcut。若 duplicates="drop" 后 bin 边少于 Q+1，
            # 则退化为 cut(rank, n_groups=有效边数-1)，labels 对应补长。
            try:
                g = pd.qcut(rk, q=Q, labels=[f"Q{i+1}" for i in range(Q)], duplicates="drop")
            except ValueError:
                # fallback：先拿 bins，再构造匹配长度的 labels
                _, bins = pd.qcut(rk, q=Q, duplicates="drop", retbins=True)
                n_eff = max(1, len(bins) - 1)
                labels_eff = [f"Q{min(i+1, Q)}" for i in range(n_eff)]
                try:
                    g = pd.qcut(rk, q=list(bins), labels=labels_eff, duplicates="drop")
                except Exception:
                    # 继续降级：用 rank 的排序分 2 组
                    g = pd.Series(np.where(rk.rank(method="first") <= len(rk) / 2, "Q1", "Q2"), index=rk.index).astype("category")
            g_df = g.dropna().astype(str).to_frame("group")
            g_df["__date__"] = d
            g_df.index.names = factor.index.names
            groups.append(g_df)
        if not groups:
            return {}, pd.DataFrame()
        groups_df = pd.concat(groups).drop(columns="__date__")

        # 计算每日换手率：每个组别留存率（今日某组里的资产还在昨日同组的比例）
        pivot_pct = {}
        pivoted = groups_df["group"].unstack(level=1)
        for q in [f"Q{i+1}" for i in range(Q)]:
            mask_today = pivoted == q
            mask_yest = mask_today.shift(1)
            # 留存率（昨日Q & 今日Q） / 昨日Q
            intersect = ((mask_today) & (mask_yest)).sum(axis=1)
            union_prev = mask_yest.sum(axis=1)
            pct = (intersect / union_prev.replace(0, np.nan)) * 100.0
            pivot_pct[q] = pct
        turnover_df = pd.DataFrame(pivot_pct)

        # 对每个 holding period，计算每组等权收益
        for hp in hp_cols:
            # fr[hp]: MultiIndex [date, asset]
            y_daily = fr[hp].dropna()
            # 对齐 group index
            aligned = groups_df[["group"]].join(y_daily.rename("y"), how="inner")
            if aligned.empty:
                continue
            # 等权收益 per date per group
            grouped = aligned.groupby([aligned.index.get_level_values(0), "group"])["y"].mean()
            # 透视为 dates × groups
            table = grouped.unstack("group").sort_index()
            # 缺失填 0（该组当日空）
            table = table.fillna(0.0)
            # 确保所有 Q 列存在：缺的补 0
            for q in [f"Q{i+1}" for i in range(Q)]:
                if q not in table.columns:
                    table[q] = 0.0
            # 多空：高减低（根据 long_quantile 调整方向）
            high_q = f"Q{Q}" if self.long_quantile == "top" else "Q1"
            low_q = "Q1" if self.long_quantile == "top" else f"Q{Q}"
            table["LS"] = table[high_q] - table[low_q]
            rets_per_hp[hp] = table
        return rets_per_hp, turnover_df

    def _alpha_decay(
        self, factor: pd.Series, ret_1d: pd.Series, lags: Sequence[int]
    ) -> pd.DataFrame:
        """
        Alpha 衰减：将因子滞后 k 期后，计算滞后因子与 ret_1d 的截面 IC/RankIC 均值。

        * lag=0  → 标准的 T 日因子 vs T 日到 T+1 日收益
        * lag=1  → T-1 日因子 vs T~T+1 日收益（看因子预测力能延续多久）
        """
        asset_level = factor.index.names[-1]
        # 先 unstack: rows=date, cols=asset
        F = factor.unstack(asset_level).sort_index()
        R = ret_1d.unstack(asset_level).sort_index().reindex(F.index)
        rows = []
        for lag in lags:
            if lag >= F.shape[0]:
                continue
            F_lag = F.shift(lag)
            # 按行算相关
            ics, rics = [], []
            for i in range(F_lag.shape[0]):
                f_i = F_lag.iloc[i].dropna()
                r_i = R.iloc[i].dropna()
                idx = f_i.index.intersection(r_i.index)
                if len(idx) < self.min_assets:
                    continue
                fi = f_i.reindex(idx).astype(float)
                ri = r_i.reindex(idx).astype(float)
                ic = float(fi.corr(ri, method="pearson"))
                ric = float(fi.corr(ri, method="spearman"))
                if not np.isnan(ic):
                    ics.append(ic)
                if not np.isnan(ric):
                    rics.append(ric)
            if not ics:
                continue
            rows.append({
                "lag": lag,
                "mean_ic": float(np.mean(ics)),
                "mean_rank_ic": float(np.mean(rics)),
                "n_dates": len(ics),
            })
        if not rows:
            return pd.DataFrame(columns=["lag", "mean_ic", "mean_rank_ic", "n_dates"]).set_index("lag")
        return pd.DataFrame(rows).set_index("lag")

    def _event_study(
        self, factor: pd.Series, fr: pd.DataFrame, hp_cols: list[str], max_h: int
    ) -> pd.DataFrame:
        """
        事件研究：在每个调仓日，做多高因子分组合、做空低因子分组合，
        持有 1..H 日的累计平均收益（事件时间坐标系）。
        """
        Q = self.quantiles
        # 每个持有期（ret_Hd）对应一个 horizon=H
        horizon_to_col = {}
        for c in hp_cols:
            try:
                H = int(c.replace("ret_", "").replace("d", ""))
                if H <= max_h:
                    horizon_to_col[H] = c
            except ValueError:
                continue
        if not horizon_to_col:
            return pd.DataFrame(columns=["avg_alpha_high_minus_low"])
        high_q = f"Q{Q}" if self.long_quantile == "top" else "Q1"
        low_q = "Q1" if self.long_quantile == "top" else f"Q{Q}"
        # 用前面实现的 _quantile_backtest 的等价逻辑：取 LS 列和各组净值
        temp_rets, _ = self._quantile_backtest(factor, fr[[horizon_to_col[h] for h in sorted(horizon_to_col)]])
        rows = []
        for H, col in sorted(horizon_to_col.items()):
            if col not in temp_rets:
                continue
            table = temp_rets[col]
            if "LS" not in table.columns:
                continue
            avg_alpha = float(table["LS"].mean())  # H 日持有累计收益的跨期平均
            rows.append({"holding_days": H, "avg_alpha_high_minus_low": avg_alpha})
        if not rows:
            return pd.DataFrame(columns=["holding_days", "avg_alpha_high_minus_low"]).set_index("holding_days")
        return pd.DataFrame(rows).set_index("holding_days").sort_index()


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------


def run_factor_tear_sheet(
    factor_panel: pd.DataFrame | pd.Series,
    forward_returns: pd.DataFrame,
    *,
    factor_name: str = "factor",
    quantiles: int = 5,
    periods_per_year: float = 365.0,
    long_quantile: Literal["top", "bottom"] = "top",
) -> FactorAnalysisResult:
    """便捷入口：一键完整评估。"""
    fa = FactorAnalyzer(quantiles=quantiles, periods_per_year=periods_per_year, long_quantile=long_quantile)
    return fa.run(factor_panel, forward_returns, factor_name=factor_name)


def build_forward_returns(
    price_panel: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
    log: bool = False,
) -> pd.DataFrame:
    """
    从 MultiIndex [date, asset] 的价格面板生成未来持有收益。

    price_panel:  MultiIndex [date, asset] × 1 列价格（例如收盘价 close）
    horizons:     持有期数（天数/根椐 frequency 理解）
    log:          True → log return，False → 几何简单收益

    返回: MultiIndex [date, asset] × [ret_1d, ret_3d, …]
    """
    s = _ensure_series(price_panel, "close").astype(float)
    asset_level = s.index.names[-1]
    P = s.unstack(asset_level).sort_index()
    out_cols = {}
    for H in horizons:
        if log:
            r = np.log(P.shift(-H) / P)
        else:
            r = P.shift(-H) / P - 1.0
        out_cols[f"ret_{H}d"] = r.stack(future_stack=True).reindex(s.index)
    return pd.DataFrame(out_cols)
