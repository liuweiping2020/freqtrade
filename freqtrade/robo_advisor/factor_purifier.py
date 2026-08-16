"""
因子预处理模块 (factor_purifier)
=================================

对标 AlphaPurify.AlphaPurifier，提供标准化的因子清洗流水线：

    缺失 → 去极值 → 中性化(行业/风格/市值) → 标准化 → (可选)PCA 正交化/降维

支持三种使用模式：
    1) 一次性快捷函数：
        clean = purify_factor_panel(factor_panel, pipeline="default", **kwargs)

    2) Pipeline 风格（可复现：fit on train, transform on test —— 避免未来信息）：
        fp = FactorPurifier(steps=[
            Winsorize(method="mad", k=5),
            Neutralize(exposures=["ln_market_cap", "industry_dummies"]),
            Standardize(method="zscore"),
        ])
        fp.fit(train_df, group=train_df.index.get_level_values("date"))
        test_clean = fp.transform(test_df)

    3) 和 Freqtrade FreqAI 对接：
        features = to_freqai_feature_dataframe(clean_factor_panel)
        可以直接作为 FreqAI 的 "feature_engineering_*" 输入

数据格式约定（与 Alphalens / Qlib 兼容）：
    factor_panel: pandas.DataFrame
        Index  : MultiIndex [date, asset]  — 截面 × 时序面板
        Columns: 一个或多个因子列
        Values : 因子原始值
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 通用工具：截面分组操作
# ---------------------------------------------------------------------------


def _group_by_date(df: pd.DataFrame) -> Iterable[tuple[Any, pd.DataFrame]]:
    """按 date 分组（无论 index 第一个 level 的名称是什么）。"""
    if isinstance(df.index, pd.MultiIndex):
        yield from df.groupby(level=0, sort=False, group_keys=False)
    else:
        # 非 MultiIndex：按整表作为单个截面
        yield 0, df


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        if df.index.nlevels < 2:
            raise ValueError("Factor DataFrame MultiIndex must have at least 2 levels [date, asset]")
        return df
    warnings.warn("Factor DataFrame is not MultiIndex, assuming single cross-section; treating index as assets.")
    return df.set_index([pd.Timestamp("today").normalize()] * len(df), append=True).swaplevel()


# ---------------------------------------------------------------------------
# Step 基类 + 内置 Steps
# ---------------------------------------------------------------------------


class PurifyStep(ABC):
    """单个预处理步骤。fit/transform 语义：fit 在训练集统计参数，transform 复用。"""

    name: str = "step"

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "PurifyStep":
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ----- 1) 缺失值处理 -----


HandleMissingMethod = Literal["drop", "ffill_crosssection", "fill_zero", "fill_median", "interpolate_time"]


class HandleMissing(PurifyStep):
    name = "handle_missing"

    def __init__(self, method: HandleMissingMethod = "fill_median", time_axis_level: str | int = 0) -> None:
        self.method = method
        self.time_axis_level = time_axis_level
        self._fill_values: dict[str, float] = {}  # fit 得到的中位数等

    def fit(self, df: pd.DataFrame) -> "HandleMissing":
        if self.method in ("fill_median",):
            self._fill_values = df.median(numeric_only=True).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.method == "drop":
            return df.dropna()
        if self.method == "fill_zero":
            return df.fillna(0.0)
        if self.method == "fill_median":
            fill = self._fill_values or df.median(numeric_only=True).to_dict()
            return df.fillna(fill)
        if self.method == "ffill_crosssection":
            # 每个截面用该截面的中位数/均值填充
            def _fill_cs(x: pd.DataFrame) -> pd.DataFrame:
                return x.fillna(x.median(numeric_only=True))
            return pd.concat([_fill_cs(x) for _, x in _group_by_date(df)]).sort_index()
        if self.method == "interpolate_time":
            # 按资产维度对时间轴线性插值
            asset_level = 1 if isinstance(df.index, pd.MultiIndex) else 0
            return (
                df.unstack(asset_level)
                .interpolate(method="linear", limit_direction="both", axis=0)
                .stack(asset_level, future_stack=True)
                .reindex(df.index)
            )
        raise ValueError(f"Unknown method {self.method!r}")


# ----- 2) 去极值（Winsorize） -----


WinsorizeMethod = Literal["mad", "sigma", "quantile", "huber"]


class Winsorize(PurifyStep):
    name = "winsorize"

    def __init__(
        self,
        method: WinsorizeMethod = "mad",
        k: float = 5.0,
        lower_q: float = 0.005,
        upper_q: float = 0.995,
        by_cross_section: bool = True,
    ) -> None:
        if method == "mad" and k <= 0:
            raise ValueError("k must be positive for mad winsorize")
        if method == "quantile" and not (0 < lower_q < upper_q < 1):
            raise ValueError("lower_q/upper_q must lie in (0,1) and lower<upper")
        self.method = method
        self.k = k
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.by_cross_section = by_cross_section
        self._global_stats: dict[str, tuple[float, float]] = {}  # 非截面模式用

    @staticmethod
    def _mad_bounds(col: pd.Series, k: float) -> tuple[float, float]:
        med = float(col.median())
        mad = float((col - med).abs().median())
        # 渐近正态一致性系数 1.4826
        sigma_hat = 1.4826 * mad if mad > 0 else float(col.std(ddof=0))
        if sigma_hat == 0:
            return med, med
        return med - k * sigma_hat, med + k * sigma_hat

    @staticmethod
    def _sigma_bounds(col: pd.Series, k: float) -> tuple[float, float]:
        mu = float(col.mean())
        s = float(col.std(ddof=0))
        if s == 0:
            return mu, mu
        return mu - k * s, mu + k * s

    def _quantile_bounds(self, col: pd.Series) -> tuple[float, float]:
        lo = float(col.quantile(self.lower_q))
        hi = float(col.quantile(self.upper_q))
        if lo == hi:
            return float(col.min()), float(col.max())
        return lo, hi

    def _huber_bounds(self, col: pd.Series, k: float = 1.5) -> tuple[float, float]:
        """Huber-style robust winsorization (IQR)."""
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return float(col.min()), float(col.max())
        return q1 - k * iqr, q3 + k * iqr

    def _per_col_bounds(self, col: pd.Series) -> tuple[float, float]:
        if self.method == "mad":
            return self._mad_bounds(col, self.k)
        if self.method == "sigma":
            return self._sigma_bounds(col, self.k)
        if self.method == "quantile":
            return self._quantile_bounds(col)
        if self.method == "huber":
            return self._huber_bounds(col, self.k)
        raise ValueError

    def fit(self, df: pd.DataFrame) -> "Winsorize":
        if not self.by_cross_section:
            stats = {}
            for c in df.columns:
                stats[c] = self._per_col_bounds(df[c].dropna())
            self._global_stats = stats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.by_cross_section:
            frames: list[pd.DataFrame] = []
            for _, xs in _group_by_date(df):
                xs = xs.copy()
                for c in xs.columns:
                    lo, hi = self._per_col_bounds(xs[c].dropna())
                    xs[c] = xs[c].clip(lo, hi)
                frames.append(xs)
            return pd.concat(frames).sort_index()
        # 全局模式
        for c in df.columns:
            if c in self._global_stats:
                lo, hi = self._global_stats[c]
                df[c] = df[c].clip(lo, hi)
        return df


# ----- 3) 中性化 (Neutralize) -----


class Neutralize(PurifyStep):
    """
    因子中性化：对每个截面做 y ~ X 的 OLS，取残差作为去暴露后的因子值。

    典型 X：
        - 行业 one-hot / 行业代码 (pd.get_dummies)
        - ln(市值)、β、动量、波动率等风格因子暴露

    mode="residual"   → y_hat = 残差（经典中性，剥离 X 的影响）
    mode="orthogonal" → 对暴露矩阵做 Gram-Schmidt 正交（PCA 风格，X 间无共线性要求更稳）
    """

    name = "neutralize"

    def __init__(
        self,
        exposures: pd.DataFrame | Sequence[str] | None = None,
        mode: Literal["residual", "orthogonal"] = "residual",
        add_intercept: bool = True,
    ) -> None:
        """
        Args:
            exposures: 如果是列名列表，则从输入 df 中取这些列作为 X；
                       如果是 DataFrame，则 index 必须与 df 对齐。
        """
        self.exposures = exposures
        self.mode = mode
        self.add_intercept = add_intercept

    @staticmethod
    def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """最小二乘残差，自动含截距；通过 QR 稳定求解。"""
        if X.shape[1] == 0:
            return y.copy()
        y_valid_mask = ~np.isnan(y)
        if not y_valid_mask.any():
            return y.copy()
        X_valid = X[y_valid_mask]
        y_valid = y[y_valid_mask]
        # 处理缺失值：对 X 列简单均值填充
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_means = np.nanmean(X_valid, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X_valid_filled = np.where(np.isnan(X_valid), col_means, X_valid)
        Q, R = np.linalg.qr(X_valid_filled, mode="reduced")
        if R.shape[0] != R.shape[1]:
            return y.copy()
        beta = np.linalg.solve(R, Q.T @ y_valid)
        residual_valid = y_valid - X_valid_filled @ beta
        residuals = np.full_like(y, np.nan)
        residuals[y_valid_mask] = residual_valid
        return residuals

    def _get_exposure_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.exposures is None:
            return pd.DataFrame(index=df.index)
        if isinstance(self.exposures, (list, tuple)):
            missing = [c for c in self.exposures if c not in df.columns]
            if missing:
                raise KeyError(f"Exposure columns not in input df: {missing}")
            return df[list(self.exposures)]
        if isinstance(self.exposures, pd.DataFrame):
            return self.exposures.reindex(df.index)
        raise TypeError("exposures must be column name list or aligned DataFrame")

    def fit(self, df: pd.DataFrame) -> "Neutralize":
        return self  # 中性化无需要保存的全局参数

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        out = df.copy()
        exp_df = self._get_exposure_df(df)
        for _, idx in _group_by_date(df):
            xs = df.loc[idx.index]
            x_exp = exp_df.loc[idx.index]
            Xraw = x_exp.to_numpy(dtype=float)
            if self.add_intercept:
                Xraw = np.hstack([Xraw, np.ones((Xraw.shape[0], 1))])
            if self.mode == "orthogonal":
                # 先对 X 做列空间的正交基，然后把因子向正交补投影
                Q, _ = np.linalg.qr(Xraw, mode="reduced")
                P = np.eye(Q.shape[0]) - Q @ Q.T
                for c in xs.columns:
                    y = xs[c].to_numpy(dtype=float)
                    mask = ~np.isnan(y)
                    if not mask.any():
                        continue
                    y[mask] = P[mask][:, mask] @ y[mask]
                    out.loc[idx.index, c] = y
            else:  # residual
                for c in xs.columns:
                    y = xs[c].to_numpy(dtype=float)
                    out.loc[idx.index, c] = self._ols_residuals(y, Xraw)
        return out


# ----- 4) 标准化 (Standardize) -----


StandardizeMethod = Literal["zscore", "rank_norm", "minmax", "rank"]


class Standardize(PurifyStep):
    name = "standardize"

    def __init__(
        self,
        method: StandardizeMethod = "zscore",
        by_cross_section: bool = True,
        eps: float = 1e-12,
        rank_apply_invnorm: bool = True,
    ) -> None:
        self.method = method
        self.by_cross_section = by_cross_section
        self.eps = eps
        self.rank_apply_invnorm = rank_apply_invnorm
        self._mean: dict[str, float] = {}
        self._std: dict[str, float] = {}
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "Standardize":
        if not self.by_cross_section:
            self._mean = df.mean(numeric_only=True).to_dict()
            self._std = df.std(numeric_only=True, ddof=0).replace(0, np.nan).to_dict()
            self._min = df.min(numeric_only=True).to_dict()
            self._max = df.max(numeric_only=True).to_dict()
        return self

    @staticmethod
    def _invnorm(rank: pd.Series) -> pd.Series:
        """逆正态变换 (Van der Waerden): rank → Φ⁻¹((r - a)/(n - 2a + 1)), a=3/8"""
        from math import erf, sqrt
        n = rank.count()
        if n < 2:
            return rank * 0.0
        a = 3.0 / 8.0
        # rank 是 1-based 的 dense rank；Blom公式
        p = (rank - a) / (n - 2 * a + 1.0)
        p = p.clip(1e-6, 1 - 1e-6)
        # 使用误差函数反函数近似 Φ⁻¹(p) = √2 · erf⁻¹(2p-1)
        def _erfinv(x):
            # Abramowitz & Stegun 近似
            a = [0.886226899, -1.645349621, 0.914624893, -0.140543331]
            b = [-2.118377725, 1.442710462, -0.329097515, 0.012229801]
            c = [-1.970840454, -1.624906493, 3.429567803, 1.641345311]
            d = [3.543889200, 1.637067800]
            y = np.where(np.abs(x) < 1, x, np.sign(x) * (1 - 1e-9))
            z = np.zeros_like(y, dtype=float)
            abs_y = np.abs(y)
            cond1 = abs_y <= 0.7
            # Case 1: |y| <= 0.7
            t1 = abs_y[cond1] ** 2
            num = (((a[3] * t1 + a[2]) * t1 + a[1]) * t1 + a[0])
            den = ((((b[3] * t1 + b[2]) * t1 + b[1]) * t1 + b[0]) * t1 + 1)
            z[cond1] = (y[cond1] * num) / den
            # Case 2: |y| > 0.7
            cond2 = ~cond1
            t2 = np.sqrt(-np.log((1 - abs_y[cond2]) / 2))
            num = (((c[3] * t2 + c[2]) * t2 + c[1]) * t2 + c[0])
            den = ((d[1] * t2 + d[0]) * t2 + 1)
            z[cond2] = np.sign(y[cond2]) * num / den
            return z
        two_p_1 = 2.0 * p.to_numpy(dtype=float) - 1.0
        return pd.Series(np.sqrt(2.0) * _erfinv(two_p_1), index=rank.index)

    def _transform_cs(self, xs: pd.DataFrame) -> pd.DataFrame:
        out = xs.copy()
        for c in xs.columns:
            col = xs[c].astype(float)
            if self.method == "zscore":
                mu = col.mean()
                s = col.std(ddof=0)
                out[c] = (col - mu) / (s if s > 0 else 1.0)
            elif self.method == "minmax":
                lo, hi = col.min(), col.max()
                r = hi - lo
                out[c] = (col - lo) / r if r > 0 else 0.0
            elif self.method == "rank":
                out[c] = col.rank(method="average", pct=False)
            elif self.method == "rank_norm":
                r = col.rank(method="average")
                out[c] = self._invnorm(r) if self.rank_apply_invnorm else r
            else:
                raise ValueError
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.by_cross_section:
            frames = [self._transform_cs(xs) for _, xs in _group_by_date(df)]
            return pd.concat(frames).sort_index()
        # 全局模式
        out = df.copy()
        for c in df.columns:
            col = df[c].astype(float)
            if self.method == "zscore":
                mu = self._mean.get(c, float(np.nanmean(col)))
                s = self._std.get(c, float(np.nanstd(col)))
                out[c] = (col - mu) / (s if s and s > 0 else 1.0)
            elif self.method == "minmax":
                lo = self._min.get(c, float(np.nanmin(col)))
                hi = self._max.get(c, float(np.nanmax(col)))
                r = hi - lo
                out[c] = (col - lo) / r if r and r > 0 else 0.0
            elif self.method == "rank_norm":
                r = col.rank(method="average")
                out[c] = self._invnorm(r) if self.rank_apply_invnorm else r
            elif self.method == "rank":
                out[c] = col.rank(method="average", pct=False)
        return out


# ----- 5) PCA 正交化与降维 -----


class PCAReduce(PurifyStep):
    """
    对因子矩阵进行 PCA 正交化（保留 n_components 主成分）。
    常见使用场景：多个高度共线因子 → 正交化后再合成。

    n_components:
        - int: 保留前 K 个
        - float (0,1): 保留累计方差解释 >= threshold 的主成分
        - "all": 保留全部（仅做旋转正交）
    """

    name = "pca_reduce"

    def __init__(
        self,
        n_components: int | float | Literal["all"] = 0.95,
        standardize_before: bool = True,
    ) -> None:
        self.n_components = n_components
        self.standardize_before = standardize_before
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._components: np.ndarray | None = None  # (n_factors, n_components)
        self.explained_variance_ratio: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "PCAReduce":
        X = df.to_numpy(dtype=float)
        mask = ~np.isnan(X).any(axis=1)
        Xfit = X[mask]
        if Xfit.shape[0] < 2:
            raise ValueError("Not enough complete rows to fit PCA")
        if self.standardize_before:
            self._mean = Xfit.mean(axis=0)
            self._std = Xfit.std(axis=0, ddof=0)
            self._std = np.where(self._std < 1e-12, 1.0, self._std)
            Xs = (Xfit - self._mean) / self._std
        else:
            self._mean = np.zeros(Xfit.shape[1])
            self._std = np.ones(Xfit.shape[1])
            Xs = Xfit
        # eig(corr / cov)
        cov = (Xs.T @ Xs) / max(Xs.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh 升序，反转成降序
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        total = eigvals.sum()
        var_ratio = eigvals / total if total > 0 else eigvals
        self.explained_variance_ratio = var_ratio
        # 确定 K
        if self.n_components == "all":
            K = len(eigvals)
        elif isinstance(self.n_components, float):
            K = int(np.searchsorted(np.cumsum(var_ratio), self.n_components)) + 1
            K = min(max(K, 1), len(eigvals))
        else:
            K = int(min(self.n_components, len(eigvals)))
        self._components = eigvecs[:, :K]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._components is None:
            raise RuntimeError("PCAReduce not fitted yet.")
        X = df.to_numpy(dtype=float)
        Xs = (X - self._mean) / self._std
        # 填充 nan → 0 以确保 matmul 稳定（PCA 对缺失不友好，建议上游先 HandleMissing）
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        scores = Xs @ self._components  # (n_samples, K)
        cols = [f"PC{i+1}" for i in range(scores.shape[1])]
        return pd.DataFrame(scores, index=df.index, columns=cols)


# ---------------------------------------------------------------------------
# 统一 Pipeline 封装
# ---------------------------------------------------------------------------


PRESET_PIPELINES: dict[str, list[PurifyStep]] = {
    "default": [
        HandleMissing(method="fill_median"),
        Winsorize(method="mad", k=5),
        Standardize(method="zscore"),
    ],
    "strict": [
        HandleMissing(method="fill_median"),
        Winsorize(method="quantile", lower_q=0.005, upper_q=0.995),
        Neutralize(mode="residual"),
        Standardize(method="rank_norm"),
    ],
    "neural": [
        HandleMissing(method="fill_median"),
        Winsorize(method="sigma", k=3),
        Standardize(method="rank_norm"),  # 神经网络偏好接近高斯分布
    ],
}


@dataclass
class PurifySummary:
    """每一步骤的诊断摘要。"""

    step: str
    before_nan_ratio: float
    after_nan_ratio: float
    before_inf_count: int
    after_inf_count: int
    before_std: dict[str, float]
    after_std: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


class FactorPurifier:
    """
    因子清洗流水线。

    典型使用::

        fp = FactorPurifier(pipeline="default")
        # 或自定义：
        fp = FactorPurifier(steps=[
            HandleMissing(method="fill_median"),
            Winsorize(method="mad", k=5),
            Neutralize(exposures=["ln_cap", "industry"]),
            Standardize(method="zscore"),
        ])
        fp.fit(train_panel)
        clean = fp.transform(train_panel)
        clean_test = fp.transform(test_panel)   # 使用训练期参数：未来安全
        summary = fp.summary()                  # 每步诊断
    """

    def __init__(
        self,
        steps: Sequence[PurifyStep] | None = None,
        pipeline: Literal["default", "strict", "neural"] | None = None,
    ) -> None:
        if steps is None and pipeline is None:
            pipeline = "default"
        if steps is None:
            if pipeline not in PRESET_PIPELINES:
                raise ValueError(f"Unknown preset {pipeline!r}, choices: {list(PRESET_PIPELINES)}")
            # 拷贝实例避免不同 FactorPurifier 共享步骤对象状态
            self.steps: list[PurifyStep] = [self._clone_step(s) for s in PRESET_PIPELINES[pipeline]]
        else:
            self.steps = list(steps)
        self._summaries: list[PurifySummary] = []

    @staticmethod
    def _clone_step(step: PurifyStep) -> PurifyStep:
        import copy
        return copy.deepcopy(step)

    # ---- 信息 ------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        # 简单启发：所有步骤 fit 之后 PCA._components 等不会为 None
        return len(self._summaries) >= 1

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self._summaries:
            rows.append(
                {
                    "step": s.step,
                    "nan_before": round(s.before_nan_ratio, 4),
                    "nan_after": round(s.after_nan_ratio, 4),
                    "inf_before": s.before_inf_count,
                    "inf_after": s.after_inf_count,
                    "mean_std_before": float(np.mean(list(s.before_std.values()) or [0])),
                    "mean_std_after": float(np.mean(list(s.after_std.values()) or [0])),
                    "extra": s.extra,
                }
            )
        return pd.DataFrame(rows)

    # ---- 核心 ------------------------------------------------------------

    def _diagnose(self, df: pd.DataFrame) -> tuple[float, int, dict[str, float]]:
        vals = df.to_numpy(dtype=float)
        total = vals.size
        nan_ratio = np.isnan(vals).sum() / max(total, 1)
        inf_count = int(np.isinf(vals).sum())
        per_col_std = {c: float(df[c].std(ddof=0)) if df[c].notna().any() else 0.0 for c in df.columns}
        return float(nan_ratio), inf_count, per_col_std

    def fit(self, factor_panel: pd.DataFrame) -> "FactorPurifier":
        df = _ensure_multiindex(factor_panel).copy()
        current = df
        self._summaries = []
        for step in self.steps:
            b_nan, b_inf, b_std = self._diagnose(current)
            step.fit(current)
            after = step.transform(current)
            a_nan, a_inf, a_std = self._diagnose(after)
            self._summaries.append(
                PurifySummary(step=step.name,
                              before_nan_ratio=b_nan, after_nan_ratio=a_nan,
                              before_inf_count=b_inf, after_inf_count=a_inf,
                              before_std=b_std, after_std=a_std,
                              extra={"shape": list(after.shape)})
            )
            current = after
        return self

    def transform(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_multiindex(factor_panel).copy()
        current = df
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, factor_panel: pd.DataFrame) -> pd.DataFrame:
        return self.fit(factor_panel).transform(factor_panel)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def purify_factor_panel(
    factor_panel: pd.DataFrame,
    pipeline: Literal["default", "strict", "neural"] | str = "default",
    **purifier_kwargs: Any,
) -> tuple[pd.DataFrame, FactorPurifier]:
    """
    快捷入口：构建 FactorPurifier → fit_transform → 返回 (clean_df, purifier)。

    purifier_kwargs 会透传给 FactorPurifier 自定义 steps 的情况也支持：
        steps=[...]（此时 pipeline 参数被忽略）
    """
    steps = purifier_kwargs.pop("steps", None)
    if steps is not None:
        fp = FactorPurifier(steps=steps, **purifier_kwargs)
    else:
        fp = FactorPurifier(pipeline=pipeline, **purifier_kwargs)  # type: ignore[arg-type]
    clean = fp.fit_transform(factor_panel)
    return clean, fp


def to_freqai_feature_dataframe(clean_factor_panel: pd.DataFrame) -> pd.DataFrame:
    """
    将清洗好的 (date, asset) MultiIndex 因子面板，
    重命名列并确保可作为 FreqAI 策略特征 DataFrame 直接拼接。

    FreqAI 约定特征列前缀为 `%`，标签列前缀为 `&`。
    """
    df = clean_factor_panel.copy()
    rename = {}
    for c in df.columns:
        if not c.startswith("%"):
            rename[c] = f"%-ra_{c}"
    if rename:
        df = df.rename(columns=rename)
    return df


# ---------------------------------------------------------------------------
# 附加：缺失/异常的简要诊断
# ---------------------------------------------------------------------------


def diagnose_factor_panel(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """对因子面板逐列返回：缺失率、标准差、偏度、峰度、min/max/95%CI 范围、Inf 数。"""
    df = factor_panel
    rows = []
    for c in df.columns:
        s = df[c].astype(float)
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            rows.append({"factor": c, "n_valid": 0, "nan_ratio": 1.0})
            continue
        from scipy.stats import skew, kurtosis  # lazy import
        nan_ratio = float(s.isna().mean())
        inf_count = int(np.isinf(vals).sum())
        rows.append({
            "factor": c,
            "n_valid": vals.size,
            "nan_ratio": round(nan_ratio, 4),
            "inf_count": inf_count,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "skew": float(skew(vals)),
            "kurt": float(kurtosis(vals)),
            "min": float(np.min(vals)),
            "p01": float(np.quantile(vals, 0.01)),
            "p99": float(np.quantile(vals, 0.99)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows).set_index("factor")
