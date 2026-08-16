"""
因子合成与收益归因 (factor_tools)
==================================

FactorCombiner:
    将多个已清洗因子（同形状 × N 列面板）合成为单一复合因子：
        - equal_weight      等权
        - icir_weight       ICIR（IR=IC_mean/IC_std）滚动加权
        - symmetric_orth    对称正交：去冗余，避免重复暴露
        - learning_to_rank  ML 排名损失启发式（线性加权版本，无需额外依赖）

BarraAttributor:
    简化版 Barra 风格归因（横截面单期）：
        组合收益 = 风格暴露 × 风格因子收益 + 特质收益
        输出：allocation effect（配置）/ selection effect（选股）/ interaction（交互）
        以及按因子维度的收益分解：value / momentum / size / quality / volatility ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .factor_analyzer import build_forward_returns, run_factor_tear_sheet


# ---------------------------------------------------------------------------
# 工具：面板窗口滚动统计
# ---------------------------------------------------------------------------


def _rolling_ic(
    factor_panel: pd.DataFrame,
    forward_returns: pd.Series,
    window: int,
    method: Literal["pearson", "spearman"] = "spearman",
) -> pd.Series:
    """窗口内计算滚动 IC（截面逐日计算后再 rolling mean/std）。"""
    # 先按截面逐期算 IC
    ic_series = []
    dates_order = []
    f = factor_panel.iloc[:, 0] if isinstance(factor_panel, pd.DataFrame) else factor_panel
    asset_level = f.index.names[-1]
    F = f.unstack(asset_level).sort_index()
    R = forward_returns.unstack(asset_level).sort_index().reindex(F.index)
    for d in F.index:
        fx = F.loc[d].dropna()
        ry = R.loc[d].dropna()
        idx = fx.index.intersection(ry.index)
        if len(idx) < 20:
            ic_series.append(np.nan)
        else:
            ic = float(fx.reindex(idx).astype(float).corr(ry.reindex(idx).astype(float), method=method))
            ic_series.append(ic)
        dates_order.append(d)
    s = pd.Series(ic_series, index=dates_order)
    return s


# ---------------------------------------------------------------------------
# FactorCombiner
# ---------------------------------------------------------------------------


@dataclass
class CombineResult:
    combined: pd.DataFrame               # MultiIndex [date, asset] × 1 列 合成因子
    weights_history: pd.DataFrame        # index=date, cols=factor_name：每期权重
    summary: dict[str, Any]


CombinationMethod = Literal["equal_weight", "icir_weight", "symmetric_orth", "ltr_weighted"]


class FactorCombiner:
    """
    多因子合成器。

    Parameters:
        method             合成方法
        rolling_window     ICIR / 正交化 的回看窗口（天数）
        warmup             训练开始前多少期用等权 fallback
        orthogonal_before  在 icir / ltr 之前先做对称正交（推荐）
    """

    def __init__(
        self,
        method: CombinationMethod = "icir_weight",
        rolling_window: int = 60,
        warmup: int = 20,
        orthogonal_before: bool = True,
    ) -> None:
        self.method = method
        self.window = rolling_window
        self.warmup = warmup
        self.orthogonal_before = orthogonal_before

    # ---- 主入口 --------------------------------------------------------

    def combine(
        self,
        factor_panel: pd.DataFrame,         # [date, asset] × K factors
        forward_returns: pd.Series,         # [date, asset] 对应持有期收益（用于ICIR/训练）
    ) -> CombineResult:
        factor_names = list(factor_panel.columns)
        K = len(factor_names)
        asset_level = factor_panel.index.names[-1]
        dates = factor_panel.index.get_level_values(0).unique().sort_values()

        # --- 1) 对称正交（可选，基于滚动窗口截面标准化后做对称正交） ---
        F = factor_panel
        if self.orthogonal_before:
            F = self._symmetric_orthogonalize(F, self.window)

        # --- 2) 方法分派 ---
        if self.method == "equal_weight":
            # 每日：(f1+...+fk)/k，标准化为同等 zscore
            combined, weights_history = self._equal(F, factor_names, dates)
        elif self.method == "symmetric_orth":
            # 已经在上文做了正交 → 再做等权（最经典的「正交后等权」合成）
            combined, weights_history = self._equal(F, factor_names, dates)
        elif self.method == "icir_weight":
            combined, weights_history = self._icir(F, factor_names, dates, forward_returns, asset_level)
        elif self.method == "ltr_weighted":
            combined, weights_history = self._ltr(F, factor_names, dates, forward_returns, asset_level)
        else:
            raise ValueError(f"Unknown method {self.method!r}")

        # --- 3) 输出摘要：最终复合因子的 IC ---
        try:
            fa = run_factor_tear_sheet(combined, forward_returns.to_frame("ret_1d"), factor_name="COMBINED",
                                       periods_per_year=365)
            ic_sum = fa.ic_summary()
            summary = {
                "combined_RankIC_mean": float(ic_sum.get("RankIC_mean", np.nan)),
                "combined_RankIC_IR": float(ic_sum.get("RankIC_IR", np.nan)),
                "weights_mean": weights_history.mean().to_dict(),
                "weights_std": weights_history.std().to_dict(),
            }
        except Exception as exc:
            summary = {"error": str(exc)}

        return CombineResult(combined=combined, weights_history=weights_history, summary=summary)

    # ---- 各方法实现 ----------------------------------------------------

    @staticmethod
    def _equal(F: pd.DataFrame, names: list[str], dates: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 先按截面 zscore 统一尺度后再平均，避免量纲不同的主导
        frames = []
        for d in dates:
            f = F.xs(d, level=0).astype(float)
            z = (f - f.mean()) / f.std(ddof=0).replace(0, np.nan).fillna(1.0)
            frames.append(pd.DataFrame({d: z.mean(axis=1)}).T.stack(future_stack=True))
        comb = pd.concat(frames).sort_index()
        comb.index.names = F.index.names
        comb.columns = ["combined"]
        # 权重历史：每期都相等
        W = pd.DataFrame(1.0 / len(names), index=dates, columns=names)
        return comb, W

    @staticmethod
    def _symmetric_orthogonalize(F: pd.DataFrame, window: int) -> pd.DataFrame:
        """对称正交：对每期截面因子，按过去 window 天估计的协方差矩阵做 PCA 白化（保留方向）。
        为避免未来信息，截面 t 用 t-1 及之前的协方差估计。"""
        dates = F.index.get_level_values(0).unique().sort_values()
        out = []
        cumul = []  # 累积到 t-1 的样本
        for d in dates:
            cur = F.xs(d, level=0).astype(float)
            if len(cumul) < max(5, window // 5):
                out.append(cur)
                cumul.append(cur.fillna(0.0).to_numpy())
                continue
            X_stack = np.vstack(cumul[-window:])  # (window*n_assets, k)
            # 去均值 + 估计协方差 + 特征分解 + 白化
            X = X_stack - np.nanmean(X_stack, axis=0)
            X = np.where(np.isnan(X), 0.0, X)
            cov = (X.T @ X) / max(X.shape[0] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-9, None)
            # 对称正交：M = V Λ^{-1/2} V^T  → X·M 保持方向且协方差≈I（Mahalanobis 白化）
            D_half = np.diag(1.0 / np.sqrt(eigvals))
            M = eigvecs @ D_half @ eigvecs.T
            cur_arr = cur.to_numpy(dtype=float)
            # 行 mean / std 归一后再白化
            cur_filled = np.where(np.isnan(cur_arr), 0.0, cur_arr)
            orth = cur_filled @ M
            # 放回 nan（保持原来 mask）
            orth = np.where(np.isnan(cur_arr), np.nan, orth)
            out_df = pd.DataFrame(orth, index=cur.index, columns=cur.columns)
            out.append(out_df)
            cumul.append(cur_filled)
        result = pd.concat(out, keys=dates, names=F.index.names).swaplevel(0, 1).sort_index()
        return result.reindex(F.index)

    def _icir(
        self,
        F: pd.DataFrame,
        names: list[str],
        dates: pd.Index,
        fwd: pd.Series,
        asset_level: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 先对每个因子算每日 IC，再滚动得到 IR
        ic_per_factor = pd.DataFrame({
            col: _rolling_ic(F[[col]], fwd, window=self.window, method="spearman")
            for col in names
        }).sort_index()
        # 滚动 mean / std
        roll_mean = ic_per_factor.rolling(self.window, min_periods=self.warmup).mean()
        roll_std = ic_per_factor.rolling(self.window, min_periods=self.warmup).std(ddof=0)
        roll_ir = (roll_mean / roll_std.replace(0, np.nan)).fillna(0.0)
        # 权重 = softmax(abs(IR)) * sign(IR)？ 或直接正权重 w = |IR|。这里采用正权重 + 乘 IC 符号
        weights_history = pd.DataFrame(index=dates, columns=names, dtype=float)
        for d in dates:
            irs = roll_ir.reindex([d]).iloc[0].to_numpy()
            if np.all(np.isnan(irs)):
                w = np.ones(len(names)) / len(names)
                weights_history.loc[d] = w
                continue
            irs_f = np.nan_to_num(irs, nan=0.0)
            # 权重 ∝ max(ir, 0) → 正向 IC 才用
            pos = np.maximum(irs_f, 0.0) + 1e-6
            w = pos / pos.sum()
            # 方向：负 IC 的因子反向纳入（用 w * sign）
            weights_history.loc[d] = w
        # 现在按日生成合成因子：先每日 zscore，然后加权和 + 乘以 sign 项
        frames = []
        sign_matrix = np.sign(np.nan_to_num(roll_ir.reindex(dates).to_numpy(), nan=0.0))
        for i, d in enumerate(dates):
            f = F.xs(d, level=0).astype(float)
            z = (f - f.mean()) / f.std(ddof=0).replace(0, np.nan).fillna(1.0)
            w = weights_history.loc[d].to_numpy(dtype=float)
            s = sign_matrix[i]
            coeff = w * s
            # 最终：Σ z_k * coeff_k（coeff 带符号，表示正 IC 则同向、负 IC 则反向）
            comb_d = z.fillna(0.0).to_numpy() @ coeff
            frames.append(pd.DataFrame({d: comb_d}).T.stack(future_stack=True))
        comb_df = pd.concat(frames).sort_index()
        comb_df.index.names = F.index.names
        comb_df.columns = ["combined"]
        return comb_df, weights_history.astype(float)

    def _ltr(
        self,
        F: pd.DataFrame,
        names: list[str],
        dates: pd.Index,
        fwd: pd.Series,
        asset_level: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Learning-to-Rank 启发式线性加权：
            每期（t - window, t - 1） 解一个最小二乘：
                minimize Σ_i Σ_j max(0, -(score_i - score_j)(r_i - r_j))^2 + λ||w||²
            等价于一个简化版 RankSVM。这里用「未来收益排序」作为伪标签，
            解 F'F w = F'y_hat，其中 y_hat 是未来收益的秩归一。
        """
        K = len(names)
        weights_history = pd.DataFrame(index=dates, columns=names, dtype=float)
        # 先把 forward_returns 透视成 [date, asset]
        R = fwd.unstack(asset_level).sort_index()
        F_piv = {c: F[c].unstack(asset_level).sort_index() for c in names}
        coef_cache = np.full(K, 1.0 / K)
        for i, d in enumerate(dates):
            if i < self.warmup:
                weights_history.loc[d] = np.full(K, 1.0 / K)
                continue
            # 收集 [t-window, t-1] 样本
            idx = dates[max(0, i - self.window): i]
            X_parts, y_parts = [], []
            for dd in idx:
                f_row = np.column_stack([F_piv[c].loc[dd].to_numpy(dtype=float) for c in names])
                r_row = R.loc[dd].to_numpy(dtype=float)
                mask = ~np.isnan(f_row).any(axis=1) & ~np.isnan(r_row)
                if mask.sum() < 20:
                    continue
                X_parts.append(f_row[mask])
                # y_hat：未来收益的归一化 rank
                rr = r_row[mask]
                rk = pd.Series(rr).rank(method="average").to_numpy(dtype=float)
                y_norm = (rk - rk.mean()) / (rk.std(ddof=0) + 1e-9)
                y_parts.append(y_norm)
            if not X_parts:
                weights_history.loc[d] = coef_cache
                continue
            X = np.vstack(X_parts)
            y = np.concatenate(y_parts)
            # L2 岭回归 (X'X + λI) w = X'y
            lam = 1e-3 * max(X.shape[0], 1)
            A = X.T @ X + lam * np.eye(K)
            b = X.T @ y
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = coef_cache
            # 正权重归一
            w_abs = np.abs(w) + 1e-9
            w_norm = w_abs / w_abs.sum()
            coef_cache = w_norm.copy()
            weights_history.loc[d] = w_norm
        # 生成合成因子：每日 zscore → 加权 sum（× sign 方向）
        frames = []
        last_W = np.full(K, 1.0 / K)
        for i, d in enumerate(dates):
            f = F.xs(d, level=0).astype(float)
            z = (f - f.mean()) / f.std(ddof=0).replace(0, np.nan).fillna(1.0)
            W_i = weights_history.loc[d].to_numpy(dtype=float)
            if np.isnan(W_i).any():
                W_i = last_W
            last_W = W_i
            comb_d = z.fillna(0.0).to_numpy() @ W_i
            frames.append(pd.DataFrame({d: comb_d}).T.stack(future_stack=True))
        comb_df = pd.concat(frames).sort_index()
        comb_df.index.names = F.index.names
        comb_df.columns = ["combined"]
        return comb_df, weights_history.astype(float)


# ---------------------------------------------------------------------------
# Barra 简化收益归因
# ---------------------------------------------------------------------------


@dataclass
class BarraAttributionResult:
    period: Any
    total_return: float
    factor_returns: dict[str, float]     # 每个风格因子的总贡献
    specific_return: float               # 特质收益
    # Brinson 分解（总层面）
    allocation_effect: float             # 配置效应（行业/大类维度择时）
    selection_effect: float              # 选股效应
    interaction_effect: float            # 交互效应
    table: pd.DataFrame                  # 按资产的明细分解（w_port, w_bench, r_port, r_bench, ...）


class BarraAttributor:
    """
    简化版 Barra + Brinson 收益归因：
        - 输入：组合权重 w_p、基准权重 w_b、各资产收益 r、
                风格暴露 X（n_assets × n_factors）
        - 分解：总超额收益 = 配置效应 + 选股效应 + 交互效应（Brinson）
                = Σ_f (X'w_p - X'w_b) * f_ret  +  特质收益 （Barra）

    典型用法（截面单期）::

        ba = BarraAttributor()
        res = ba.attribute(
            portfolio_weights={'BTC':0.5,'ETH':0.3,'BNB':0.2},
            benchmark_weights={'BTC':0.45,'ETH':0.35,'BNB':0.2},
            returns={'BTC':0.02,'ETH':-0.01,'BNB':0.005},
            style_exposures={
                'BTC': {'size':1.2,'value':-0.3,'momentum':0.8,'volatility':0.2},
                ...
            }
        )
    """

    def __init__(self, factor_fallback: str = "mean") -> None:
        self.factor_fallback = factor_fallback

    def attribute(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        returns: dict[str, float],
        style_exposures: dict[str, dict[str, float]] | None = None,
        period: Any = None,
    ) -> BarraAttributionResult:
        syms = list(set(portfolio_weights) | set(benchmark_weights) | set(returns))
        wp = np.array([portfolio_weights.get(s, 0.0) for s in syms], dtype=float)
        wb = np.array([benchmark_weights.get(s, 0.0) for s in syms], dtype=float)
        r = np.array([returns.get(s, 0.0) for s in syms], dtype=float)

        total_port = float(wp @ r)
        total_bench = float(wb @ r)
        active_total = total_port - total_bench

        # --- Brinson 分解 ---
        # allocation (择时) : Σ (wp_i - wb_i) * (rb_i - rb_total)   行业配置 → 这里按资产级简化为上式（近似）
        # selection  (选股) : Σ wb_i * (rp_i - rb_i)  → 若个股资产级别 rp=rb=returns，selection=0。
        #                    我们退化为：选择了某资产而基准没选的贡献。
        # interaction        : Σ (wp_i - wb_i) * (rp_i - rb_i)
        #
        # 简单实现：令 r_bench = r， Brinson 展开为：
        #   allocation = Σ (wp - wb) * (r - total_bench)
        #   selection  = Σ wb * (r - r) ≡ 0 （个股级无选股差异，恒为 0）
        #   超额 = allocation + interaction  → 因此我们把 interaction 拆出来
        allocation = float(np.sum((wp - wb) * (r - total_bench)))
        interaction = float(np.sum((wp - wb) * (r - total_bench)))
        # 让三者和等于 active_total：
        # 定义 selection = active_total - allocation - interaction
        selection = active_total - allocation - interaction

        # --- Barra 风格归因（若提供风格暴露）---
        factor_returns: dict[str, float] = {}
        specific_return = 0.0
        if style_exposures:
            factor_names = []
            for s in syms:
                for k in style_exposures.get(s, {}).keys():
                    if k not in factor_names:
                        factor_names.append(k)
            if factor_names:
                X = np.zeros((len(syms), len(factor_names)), dtype=float)
                for i, s in enumerate(syms):
                    for j, f in enumerate(factor_names):
                        X[i, j] = style_exposures.get(s, {}).get(f, 0.0)
                # 截面 Fama-MacBeth 回归：r_i = Σ X_ij f_j + u_i  → 得到 f 及 u
                # 加上截距（市场因子）
                Xa = np.hstack([X, np.ones((X.shape[0], 1))])
                try:
                    beta, *_ = np.linalg.lstsq(Xa, r, rcond=None)
                except Exception:
                    beta = np.zeros(Xa.shape[1])
                f_ret = beta[:-1]
                alpha = beta[-1]
                u = r - Xa @ beta
                # 组合层面风格贡献 = (X'wp - X'wb) · f + α*(wp-wb)
                active_X = X.T @ (wp - wb)
                for name, fx in zip(factor_names, active_X * f_ret):
                    factor_returns[name] = float(fx)
                specific_return = float(np.sum((wp - wb) * u) + alpha * (wp - wb).sum())

        table = pd.DataFrame(
            {
                "w_port": wp.round(6),
                "w_bench": wb.round(6),
                "w_diff": (wp - wb).round(6),
                "return": r.round(6),
                "contrib_port": (wp * r).round(6),
                "contrib_bench": (wb * r).round(6),
                "contrib_diff": ((wp - wb) * r).round(6),
            },
            index=syms,
        )
        return BarraAttributionResult(
            period=period,
            total_return=active_total,
            factor_returns=factor_returns,
            specific_return=specific_return,
            allocation_effect=allocation,
            selection_effect=selection,
            interaction_effect=interaction,
            table=table,
        )
