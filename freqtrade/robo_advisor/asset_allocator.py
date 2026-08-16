"""
资产配置引擎 (asset_allocator)
===============================

实现三类主流配置模型：
    1. MarkowitzAllocator        均值-方差（马科维茨）：最大夏普 / 最小波动 / 目标收益 / 目标波动
    2. RiskParityAllocator       风险平价（等风险贡献 ERC）：每个资产对组合风险贡献相等
    3. BlackLittermanAllocator   Black-Litterman：均衡收益 + 主观观点 → 后验收益 → 优化

统一 API:
    allocator.allocate(
        assets,                    # 资产名列表
        expected_returns,          # 期望收益率 (n,) 或 None
        cov_matrix,                # 协方差矩阵 (n, n)
        risk_free_rate=0.0,
        max_weight=None,           # 单资产上限
        min_weight=None,           # 单资产下限（可 0，若允许做空可为负）
        budget=1.0,                # 权重和约束（一般 1.0）
        target_volatility=None,    # 目标波动率（Markowitz 目标波动模式）
        target_return=None,        # 目标收益率（Markowitz 目标收益模式）
        ...
    ) -> AllocationResult

AllocateForProfile 便捷入口：
    allocate_for_profile(profile, assets, returns_df_or_cov) → AllocationResult
    * 根据 InvestorProfile 自动设置：
        - 权重上限（保守→更分散；激进→允许集中）
        - 允许做空与否（R4 以下一般不允许做空）
        - 目标波动率（基于 max_drawdown_tolerance 推导）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import chi2

from .user_profile import InvestorProfile, RiskLevel


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """资产配置结果（统一输出，便于前端展示 & 回测）。"""

    assets: list[str]
    weights: np.ndarray                              # (n,)
    expected_return: float                           # 年化预期收益（基于模型假设）
    volatility: float                                # 年化波动率（组合标准差）
    sharpe: float                                    # 夏普比率 (μ - rf) / σ
    risk_contributions: np.ndarray                   # (n,) 各资产边际风险贡献 * 权重，和 = σ²
    diversify_ratio: float                           # 分散率 = weighted_avg_σ / σ，>1 说明有分散化收益
    turnover: float = 0.0                            # 相对于 prior weights 的换手率（如有）
    model: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "asset": self.assets,
                "weight": self.weights.round(6),
                "risk_contrib": self.risk_contributions.round(6),
            }
        )
        df["risk_contrib_pct"] = np.where(
            self.volatility > 0,
            (df["risk_contrib"] / (self.volatility**2)).round(4),
            0.0,
        )
        return df.set_index("asset")

    @property
    def weight_dict(self) -> dict[str, float]:
        return dict(zip(self.assets, np.round(self.weights, 6).tolist()))

    def summary(self) -> dict[str, float]:
        return {
            "expected_return": round(self.expected_return, 4),
            "volatility": round(self.volatility, 4),
            "sharpe": round(self.sharpe, 4),
            "diversify_ratio": round(self.diversify_ratio, 4),
            "turnover": round(self.turnover, 4),
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _as_arrays(
    assets: Sequence[str],
    expected_returns: Sequence[float] | np.ndarray | None,
    cov_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    asset_list = list(assets)
    n = len(asset_list)
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.shape != (n, n):
        raise ValueError(f"cov_matrix shape mismatch: {cov.shape} vs n={n}")
    if expected_returns is None:
        mu = np.zeros(n, dtype=float)
    else:
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)
        if mu.shape[0] != n:
            raise ValueError(f"expected_returns length mismatch: {mu.shape[0]} vs n={n}")
    return asset_list, mu, cov


def _portfolio_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float
) -> tuple[float, float, float]:
    port_mu = float(weights @ mu)
    port_var = float(weights @ cov @ weights)
    port_sigma = float(np.sqrt(max(port_var, 0.0)))
    sharpe = float((port_mu - rf) / port_sigma) if port_sigma > 0 else 0.0
    return port_mu, port_sigma, sharpe


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """欧拉分解：RC_i = w_i * (Σw)_i，sum = w'Σw = σ²"""
    sigma_p_sq = float(weights @ cov @ weights)
    if sigma_p_sq <= 0:
        return np.zeros_like(weights)
    # 边际风险贡献
    mrc = cov @ weights  # (n,)
    rc = weights * mrc    # (n,)
    return rc


def _diversify_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    port_sigma = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    if port_sigma == 0:
        return 1.0
    sigmas = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    weighted_avg_sigma = float(np.abs(weights) @ sigmas)
    return weighted_avg_sigma / port_sigma if port_sigma > 0 else 1.0


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class BaseAllocator(ABC):
    """资产配置器基类，统一约束接口。"""

    model_name: str = "base"

    def __init__(
        self,
        *,
        allow_short: bool = False,
        default_max_weight: float | None = None,
        default_min_weight: float | None = None,
    ) -> None:
        self.allow_short = allow_short
        self.default_max_weight = default_max_weight
        self.default_min_weight = default_min_weight if default_min_weight is not None else (
            -1.0 if allow_short else 0.0
        )

    # --- 对外入口 ---------------------------------------------------------

    @abstractmethod
    def allocate(
        self,
        assets: Sequence[str],
        expected_returns: Sequence[float] | np.ndarray | None,
        cov_matrix: Sequence[Sequence[float]] | np.ndarray,
        *,
        risk_free_rate: float = 0.0,
        max_weight: float | None = None,
        min_weight: float | None = None,
        per_asset_max: dict[str, float] | None = None,
        per_asset_min: dict[str, float] | None = None,
        budget: float = 1.0,
        group_constraints: list[dict[str, Any]] | None = None,
        prior_weights: np.ndarray | None = None,
        turnover_limit: float | None = None,
        **kwargs: Any,
    ) -> AllocationResult:
        ...

    # --- 辅助：构建约束与边界 -------------------------------------------

    def _build_bounds(
        self,
        n: int,
        assets: list[str],
        max_weight: float | None,
        min_weight: float | None,
        per_asset_max: dict[str, float] | None,
        per_asset_min: dict[str, float] | None,
    ) -> Bounds:
        lo = np.full(n, self.default_min_weight, dtype=float)
        hi = np.full(n, self.default_max_weight if self.default_max_weight is not None else 1.0, dtype=float)
        if min_weight is not None:
            lo[:] = min_weight
        if max_weight is not None:
            hi[:] = max_weight
        if per_asset_min:
            for i, a in enumerate(assets):
                if a in per_asset_min:
                    lo[i] = per_asset_min[a]
        if per_asset_max:
            for i, a in enumerate(assets):
                if a in per_asset_max:
                    hi[i] = per_asset_max[a]
        return Bounds(lo, hi, keep_feasible=True)

    @staticmethod
    def _build_constraints(
        n: int,
        budget: float,
        group_constraints: list[dict[str, Any]] | None,
        assets: list[str],
        prior_weights: np.ndarray | None,
        turnover_limit: float | None,
    ) -> list[Any]:
        cons: list[Any] = []
        # 权重和 = budget
        cons.append(LinearConstraint(np.ones((1, n)), lb=budget, ub=budget))
        # 分组约束：[{"assets":["BTC","ETH"],"min":0.3,"max":0.5}, ...]
        if group_constraints:
            for gc in group_constraints:
                vec = np.zeros(n)
                for a in gc["assets"]:
                    if a in assets:
                        vec[assets.index(a)] = 1.0
                lb = gc.get("min", -np.inf)
                ub = gc.get("max", np.inf)
                if not np.isneginf(lb) or not np.isposinf(ub):
                    cons.append(LinearConstraint(vec.reshape(1, n), lb=lb, ub=ub))
        # 换手约束：sum(|w - w0|) <= turnover_limit
        if prior_weights is not None and turnover_limit is not None:
            # 通过引入辅助变量实现。为了接口简洁，此处采用 2*n 约束的 slack trick：
            # 对 ∑|Δ| ≤ L，我们用线性约束： -L ≤ Σ (w - w0) ≤ L 是近似的；
            # 但准确做法需二次规划扩展变量。在本模块用一对带符号和来近似是不够的，
            # 因此：在 scipy SLSQP 框架下通过 penalty 方法近似处理（在目标函数加罚项）。
            # 这里不添加 LinearConstraint，相关处理放在各 allocate() 的 penalty。
            _ = (prior_weights, turnover_limit)  # silence
        return cons


# ---------------------------------------------------------------------------
# 1) Markowitz 均值-方差
# ---------------------------------------------------------------------------


MarkowitzObjective = Literal["max_sharpe", "min_volatility", "target_return", "target_volatility", "max_return"]


class MarkowitzAllocator(BaseAllocator):
    """
    均值-方差（马科维茨）资产配置。

    优化目标(objective)：
        - max_sharpe        : 最大化 (μ - rf) / σ   [默认]
        - min_volatility    : 最小化组合方差
        - target_return     : 在 μ >= target_return 下最小化方差
        - target_volatility : 在 σ <= target_vol 下最大化收益
        - max_return        : 最大化预期收益（风险不约束，通常只在有上限约束时才有解）
    """

    model_name = "markowitz"

    def __init__(self, objective: MarkowitzObjective = "max_sharpe", **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self.objective = objective

    # ------------------------------------------------------------------
    def allocate(  # noqa: D401 (complex signature by design)
        self,
        assets: Sequence[str],
        expected_returns: Sequence[float] | np.ndarray | None,
        cov_matrix: Sequence[Sequence[float]] | np.ndarray,
        *,
        risk_free_rate: float = 0.0,
        max_weight: float | None = None,
        min_weight: float | None = None,
        per_asset_max: dict[str, float] | None = None,
        per_asset_min: dict[str, float] | None = None,
        budget: float = 1.0,
        group_constraints: list[dict[str, Any]] | None = None,
        prior_weights: np.ndarray | None = None,
        turnover_limit: float | None = None,
        target_return: float | None = None,
        target_volatility: float | None = None,
        solver: str = "SLSQP",
        maxiter: int = 1000,
        tol: float = 1e-10,
        **_: Any,
    ) -> AllocationResult:
        asset_list, mu, cov = _as_arrays(assets, expected_returns, cov_matrix)
        n = len(asset_list)

        bounds = self._build_bounds(n, asset_list, max_weight, min_weight, per_asset_max, per_asset_min)
        constraints = self._build_constraints(n, budget, group_constraints, asset_list, prior_weights, turnover_limit)

        # 目标 & 附加约束
        obj = self.objective
        extra_cons: list[Any] = []
        x0 = np.full(n, budget / n)

        if obj == "min_volatility":
            def f(w):
                return float(w @ cov @ w)
        elif obj == "max_sharpe":
            if not np.any(mu):
                raise ValueError("max_sharpe requires non-zero expected_returns")
            # 最大化夏普等价于最小化 -sharpe。加上数值安全。
            def f(w):
                pvar = float(w @ cov @ w)
                psig = np.sqrt(max(pvar, 1e-18))
                pret = float(w @ mu)
                return -(pret - risk_free_rate) / (psig + 1e-12)
        elif obj == "target_return":
            if target_return is None:
                raise ValueError("target_return required for objective 'target_return'")
            # 期望收益约束: mu'w >= target_return
            extra_cons.append(LinearConstraint(mu.reshape(1, n), lb=target_return, ub=np.inf))
            def f(w):
                return float(w @ cov @ w)
        elif obj == "target_volatility":
            if target_volatility is None:
                raise ValueError("target_volatility required for objective 'target_volatility'")
            tv2 = target_volatility ** 2
            # 非线性约束：w'cov w <= tv2
            def vol_ineq(w):
                return tv2 - float(w @ cov @ w)  # >= 0
            extra_cons.append({"type": "ineq", "fun": vol_ineq})
            def f(w):
                return -float(w @ mu)  # 最大化收益 → 最小化负收益
        elif obj == "max_return":
            def f(w):
                return -float(w @ mu)
        else:
            raise ValueError(f"Unknown objective {obj!r}")

        # 换手惩罚（近似）
        if prior_weights is not None and turnover_limit is not None:
            base_f = f
            penalty_scale = 1e3
            def f_pen(w):
                to = float(np.sum(np.abs(w - prior_weights)))
                extra = penalty_scale * max(0.0, to - turnover_limit) ** 2
                return base_f(w) + extra
            f = f_pen

        res = minimize(
            f,
            x0,
            method=solver,
            bounds=bounds,
            constraints=constraints + extra_cons,
            options={"maxiter": maxiter, "ftol": tol, "disp": False},
        )
        if not res.success:
            # fallback：等权
            w_final = np.full(n, budget / n)
        else:
            w_final = np.asarray(res.x, dtype=float)
        # 归一（保证数值误差下预算仍然满足）
        s = w_final.sum()
        if abs(s) > 1e-12:
            w_final = w_final / s * budget
        # 数值裁剪到边界内
        lb_arr = np.asarray(bounds.lb)
        ub_arr = np.asarray(bounds.ub)
        w_final = np.clip(w_final, lb_arr, ub_arr)

        pret, pvol, psharpe = _portfolio_stats(w_final, mu, cov, risk_free_rate)
        rc = _risk_contributions(w_final, cov)
        dr = _diversify_ratio(w_final, cov)
        to = float(np.sum(np.abs(w_final - prior_weights))) if prior_weights is not None else 0.0

        return AllocationResult(
            assets=asset_list,
            weights=w_final,
            expected_return=pret,
            volatility=pvol,
            sharpe=psharpe,
            risk_contributions=rc,
            diversify_ratio=dr,
            turnover=to,
            model=f"{self.model_name}:{self.objective}",
            meta={"success": bool(getattr(res, "success", False)), "message": str(getattr(res, "message", ""))},
        )


# ---------------------------------------------------------------------------
# 2) 风险平价（等风险贡献 ERC）
# ---------------------------------------------------------------------------


class RiskParityAllocator(BaseAllocator):
    """
    风险平价（Risk Parity）—— 等风险贡献（Equal Risk Contribution, ERC）。

    优化思路：最小化 Σ_i Σ_j (RC_i - RC_j)²，约束 Σw = 1, w ≥ 0（或 allow_short）。
    对小规模问题（n ≤ 50）收敛很快。
    """

    model_name = "risk_parity"

    def allocate(
        self,
        assets: Sequence[str],
        expected_returns: Sequence[float] | np.ndarray | None,
        cov_matrix: Sequence[Sequence[float]] | np.ndarray,
        *,
        risk_free_rate: float = 0.0,
        max_weight: float | None = None,
        min_weight: float | None = None,
        per_asset_max: dict[str, float] | None = None,
        per_asset_min: dict[str, float] | None = None,
        budget: float = 1.0,
        group_constraints: list[dict[str, Any]] | None = None,
        prior_weights: np.ndarray | None = None,
        turnover_limit: float | None = None,
        solver: str = "SLSQP",
        maxiter: int = 2000,
        tol: float = 1e-12,
        **_: Any,
    ) -> AllocationResult:
        asset_list, mu, cov = _as_arrays(assets, expected_returns, cov_matrix)
        n = len(asset_list)

        # 保证协方差是 PSD（略加对角阻尼）
        cov = (cov + cov.T) / 2.0
        cov += np.eye(n) * 1e-10

        bounds = self._build_bounds(n, asset_list, max_weight, min_weight, per_asset_max, per_asset_min)
        # 风险平价要求 w ≥ 0（经典定义）；如果 allow_short=True 仍会退化为非负较稳定

        constraints = self._build_constraints(n, budget, group_constraints, asset_list, prior_weights, turnover_limit)

        def objective(w):
            # 目标：Σ_i Σ_j (RC_i - RC_j)² → 0 等价于所有 RC_i 相等
            rc = _risk_contributions(w, cov)
            # outer 差的平方和
            diff = rc.reshape(-1, 1) - rc.reshape(1, -1)
            return float(np.sum(diff ** 2))

        # 初值：按 1/σ 比例启发式
        sigmas = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        inv_sig = 1.0 / sigmas
        x0 = (inv_sig / inv_sig.sum()) * budget
        x0 = np.clip(x0, bounds.lb, bounds.ub)

        res = minimize(
            objective,
            x0,
            method=solver,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": tol, "disp": False},
        )
        w_final = np.asarray(res.x if res.success else x0, dtype=float)
        s = w_final.sum()
        if abs(s) > 1e-12:
            w_final = w_final / s * budget
        w_final = np.clip(w_final, bounds.lb, bounds.ub)

        pret, pvol, psharpe = _portfolio_stats(w_final, mu, cov, risk_free_rate)
        rc = _risk_contributions(w_final, cov)
        dr = _diversify_ratio(w_final, cov)
        to = float(np.sum(np.abs(w_final - prior_weights))) if prior_weights is not None else 0.0

        return AllocationResult(
            assets=asset_list,
            weights=w_final,
            expected_return=pret,
            volatility=pvol,
            sharpe=psharpe,
            risk_contributions=rc,
            diversify_ratio=dr,
            turnover=to,
            model=self.model_name,
            meta={"success": bool(res.success), "message": str(res.message), "obj": float(res.fun)},
        )


# ---------------------------------------------------------------------------
# 3) Black-Litterman
# ---------------------------------------------------------------------------


@dataclass
class BLView:
    """
    Black-Litterman 主观观点。

    Views 用线性形式表达：  P · μ ~ N(q, Ω)

    示例（4 个资产 A/B/C/D）：
        - 绝对观点："A 的年化收益期望是 8%"    → P=[1,0,0,0], q=0.08
        - 相对观点："A 比 B 高 3%"           → P=[1,-1,0,0], q=0.03
    """

    P: np.ndarray                         # (k, n) 观点载荷矩阵
    q: np.ndarray                         # (k,)   观点收益
    confidence: np.ndarray | None = None  # (k,) 观点置信度 0~1，用于构造 Ω
    tau: float = 0.05                     # 先验不确定性缩放（通常 0.025~0.3）

    @classmethod
    def from_list(cls, views: list[dict[str, Any]], n: int, assets: list[str], tau: float = 0.05) -> "BLView":
        k = len(views)
        P = np.zeros((k, n), dtype=float)
        q = np.zeros(k, dtype=float)
        conf = np.zeros(k, dtype=float)
        for i, v in enumerate(views):
            for a, w in v["assets"].items():
                if a in assets:
                    P[i, assets.index(a)] = w
            q[i] = v["return"]
            conf[i] = v.get("confidence", 0.5)
        return cls(P=P, q=q, confidence=conf, tau=tau)


class BlackLittermanAllocator(BaseAllocator):
    """
    Black-Litterman 配置：
        1) 估计市场均衡先验 π（reverse optimization 或直接传入）
        2) 结合主观观点 (P, q, Ω) → 后验收益 E[R|view]、后验协方差 M
        3) 在后验下做二次优化（默认目标：最大夏普）

    典型用法::

        bl = BlackLittermanAllocator()
        result = bl.allocate(
            assets=["BTC","ETH","BNB","SOL"],
            expected_returns=None,        # 若传入，则作为均衡 π 使用；否则由 market_weights 反推
            cov_matrix=cov,
            market_weights={"BTC":0.6,"ETH":0.25,"BNB":0.1,"SOL":0.05},
            views=[
                {"assets": {"BTC":1}, "return": 0.15, "confidence": 0.7},
                {"assets": {"SOL":1, "ETH":-1}, "return": 0.08, "confidence": 0.5},
            ],
            objective="max_sharpe",
        )
    """

    model_name = "black_litterman"

    def __init__(self, objective: MarkowitzObjective = "max_sharpe", **base_kwargs: Any) -> None:
        super().__init__(**base_kwargs)
        self._mv = MarkowitzAllocator(objective=objective, **base_kwargs)

    # ------------------------------------------------------------------
    @staticmethod
    def reverse_optimize(
        market_weights: np.ndarray, cov: np.ndarray, rf: float, delta: float | None = None
    ) -> np.ndarray:
        """
        反向优化（reverse optimization）：
            π = δ Σ w_mkt ，其中 δ 为风险厌恶系数（默认 2.5）
        """
        if delta is None:
            # 经验式：假设组合夏普 ~1，令 δ = 夏普/σ_p；若未知则默认 2.5
            delta = 2.5
        return rf + delta * (cov @ market_weights)

    @staticmethod
    def compute_posterior(
        pi: np.ndarray, cov: np.ndarray, view: BLView
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        返回后验期望收益 μ_BL 与后验协方差（收缩形式，He & Litterman 1999）::

            M = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
            μ_BL = M [(τΣ)⁻¹ π + P'Ω⁻¹ q]
        """
        n = len(pi)
        tau = view.tau
        tau_sigma = tau * cov
        # 构造 Ω：对角矩阵 Ω_ii = (P_i · Σ · P_i') * tau  （默认 Idzorek 由 confidence 调整）
        k = view.P.shape[0]
        omega = np.zeros((k, k), dtype=float)
        for i in range(k):
            p_i = view.P[i]
            base_uncertainty = float(p_i @ tau_sigma @ p_i)
            if view.confidence is not None:
                # 置信度 c → 缩放：uncertainty = base / c^2 （简单启发式，c=1 时与市场精度相符）
                c = max(view.confidence[i], 1e-3)
                omega[i, i] = base_uncertainty / (c ** 2)
            else:
                omega[i, i] = base_uncertainty

        tau_sigma_inv = np.linalg.pinv(tau_sigma + 1e-6 * np.eye(n))
        omega_inv = np.linalg.pinv(omega + 1e-9 * np.eye(k))
        M_inv = tau_sigma_inv + view.P.T @ omega_inv @ view.P
        M = np.linalg.pinv(M_inv + 1e-9 * np.eye(n))
        mu_bl = M @ (tau_sigma_inv @ pi + view.P.T @ omega_inv @ view.q)
        # 后验协方差（用于后续优化）：M + Σ（标准形式）
        post_cov = cov + M
        post_cov = (post_cov + post_cov.T) / 2.0
        return mu_bl, post_cov

    # ------------------------------------------------------------------
    def allocate(
        self,
        assets: Sequence[str],
        expected_returns: Sequence[float] | np.ndarray | None,
        cov_matrix: Sequence[Sequence[float]] | np.ndarray,
        *,
        risk_free_rate: float = 0.0,
        max_weight: float | None = None,
        min_weight: float | None = None,
        per_asset_max: dict[str, float] | None = None,
        per_asset_min: dict[str, float] | None = None,
        budget: float = 1.0,
        group_constraints: list[dict[str, Any]] | None = None,
        prior_weights: np.ndarray | None = None,
        turnover_limit: float | None = None,
        # BL 特有参数
        market_weights: dict[str, float] | np.ndarray | None = None,
        views: list[dict[str, Any]] | BLView | None = None,
        delta: float | None = None,
        tau: float = 0.05,
        target_return: float | None = None,
        target_volatility: float | None = None,
        solver: str = "SLSQP",
        maxiter: int = 1500,
        **_: Any,
    ) -> AllocationResult:
        asset_list, _mu_in, cov = _as_arrays(assets, expected_returns, cov_matrix)
        n = len(asset_list)

        # --- 1) 先验均衡收益 π ---
        if expected_returns is not None:
            pi = _mu_in
        elif market_weights is not None:
            if isinstance(market_weights, dict):
                wm = np.array([market_weights.get(a, 0.0) for a in asset_list], dtype=float)
            else:
                wm = np.asarray(market_weights, dtype=float)
            s = wm.sum()
            if abs(s) < 1e-12:
                raise ValueError("market_weights sum to 0")
            wm = wm / s
            pi = self.reverse_optimize(wm, cov, risk_free_rate, delta)
        else:
            raise ValueError("BlackLittermanAllocator requires expected_returns OR market_weights")

        # --- 2) 观点 → 后验 ---
        if views is None:
            mu_bl, post_cov = pi, cov
        elif isinstance(views, BLView):
            mu_bl, post_cov = self.compute_posterior(pi, cov, views)
        else:
            blv = BLView.from_list(views, n, asset_list, tau=tau)
            mu_bl, post_cov = self.compute_posterior(pi, cov, blv)

        # --- 3) 后验下用 Markowitz 求最优权重 ---
        result = self._mv.allocate(
            asset_list,
            mu_bl,
            post_cov,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
            min_weight=min_weight,
            per_asset_max=per_asset_max,
            per_asset_min=per_asset_min,
            budget=budget,
            group_constraints=group_constraints,
            prior_weights=prior_weights,
            turnover_limit=turnover_limit,
            target_return=target_return,
            target_volatility=target_volatility,
            solver=solver,
            maxiter=maxiter,
        )
        # 记录 BL 特有信息
        result.model = self.model_name
        result.meta.update({"prior_pi": pi, "posterior_mu": mu_bl})
        return result


# ---------------------------------------------------------------------------
# 快捷入口：根据 InvestorProfile 自动约束配置
# ---------------------------------------------------------------------------


def allocate_for_profile(
    profile: InvestorProfile,
    assets: Sequence[str],
    cov_matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_returns: Sequence[float] | np.ndarray | None = None,
    model: Literal["markowitz", "risk_parity", "black_litterman"] = "markowitz",
    objective: MarkowitzObjective = "max_sharpe",
    use_asset_mix_prior: bool = True,
    views: list[dict[str, Any]] | None = None,
    market_weights: dict[str, float] | None = None,
    risk_free_rate: float = 0.02,
    budget: float = 1.0,
    **extra_kwargs: Any,
) -> AllocationResult:
    """
    根据 InvestorProfile 自动构建合理的约束，然后调用对应 allocator。

    自动映射策略：
        max_weight           R1=0.30  R2=0.40  R3=0.55  R4=0.75  R5=0.95
        min_weight           0 (default)
        target_volatility    由 max_drawdown_tolerance 启发： σ ≈ |dd| / 2 （年频下近似）
    """
    rl = profile.risk_level
    max_w_map = {RiskLevel.R1: 0.30, RiskLevel.R2: 0.40, RiskLevel.R3: 0.55, RiskLevel.R4: 0.75, RiskLevel.R5: 0.95}
    max_w = max_w_map[rl]

    # 根据画像设置 per_asset 先验（可选）
    per_asset_max = extra_kwargs.pop("per_asset_max", None)
    per_asset_min = extra_kwargs.pop("per_asset_min", None)

    # 目标波动（只在目标波动模式或给 Markowitz 启发）
    # 经验式：最大回撤 ≈ 2 × σ （按年度化），因此 σ ≈ |dd| / 2
    suggested_vol = abs(profile.max_drawdown_tolerance) / 2.0

    kwargs: dict[str, Any] = dict(
        assets=assets,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        max_weight=max_w,
        budget=budget,
        per_asset_max=per_asset_max,
        per_asset_min=per_asset_min,
    )
    kwargs.update(extra_kwargs)

    if model == "markowitz":
        alloc = MarkowitzAllocator(objective=objective, allow_short=(rl >= RiskLevel.R4))
        if objective == "target_volatility":
            kwargs.setdefault("target_volatility", suggested_vol)
    elif model == "risk_parity":
        alloc = RiskParityAllocator(allow_short=False)
    elif model == "black_litterman":
        alloc = BlackLittermanAllocator(objective=objective, allow_short=(rl >= RiskLevel.R4))
        if views is not None:
            kwargs["views"] = views
        if market_weights is not None:
            kwargs["market_weights"] = market_weights
        if expected_returns is None and market_weights is None and use_asset_mix_prior:
            # 用画像建议的 mix 作为均衡权重 fallback
            mix = profile.suggest_asset_mix()
            # 仅对匹配的资产给权重，其余给 0
            default_mw = {a: mix.get(a, 0.0) for a in assets}
            s = sum(default_mw.values())
            if s > 0:
                default_mw = {k: v / s for k, v in default_mw.items()}
            kwargs["market_weights"] = default_mw
        if objective == "target_volatility":
            kwargs.setdefault("target_volatility", suggested_vol)
    else:
        raise ValueError(f"Unknown model {model!r}")

    result = alloc.allocate(**kwargs)
    result.meta.setdefault("suggested_volatility", suggested_vol)
    result.meta.setdefault("profile_risk_level", rl.name)
    return result


# ---------------------------------------------------------------------------
# 附加：从收益率 DataFrame 估计协方差与期望收益
# ---------------------------------------------------------------------------


def estimate_covariance(
    returns_df: pd.DataFrame,
    *,
    method: Literal["sample", "ledoit_wolf", "shrinkage"] = "ledoit_wolf",
    annualize: int | None = 365,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    从收益率面板 (T × n) 估计期望收益与协方差矩阵。

    Args:
        returns_df:  日/小时收益率 DataFrame（index=日期，columns=资产名）
        method:      sample（样本）/ ledoit_wolf（Ledoit-Wolf 线性收缩）
        annualize:   年化乘数（日频→365，小时频→365*24…）；None 表示不年化
    Returns:
        mu (n,), cov (n,n), assets (list[str])
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be pandas.DataFrame")
    assets = list(returns_df.columns)
    X = returns_df.to_numpy(dtype=float)
    mu_s = np.nanmean(X, axis=0)
    Xc = X - mu_s
    # 删缺失行
    mask = ~np.isnan(Xc).any(axis=1)
    Xc = Xc[mask]
    T = Xc.shape[0]
    if T < 2:
        raise ValueError("Not enough observations for covariance estimation")
    sample = (Xc.T @ Xc) / (T - 1)

    if method == "sample":
        cov = sample
    else:
        # Ledoit-Wolf 收缩（单因子目标：对角矩阵，方差均值）
        n = sample.shape[0]
        avg_var = np.trace(sample) / n
        target = avg_var * np.eye(n)
        # 收缩强度 δ 的简化估计
        # （精确 LW 有解析公式，这里使用一个简化稳健近似：随 n/T 递增收缩）
        shrink = np.clip(n / (T + n), 0.05, 0.8) if method in ("ledoit_wolf", "shrinkage") else 0.2
        cov = (1 - shrink) * sample + shrink * target

    if annualize:
        mu_s = mu_s * annualize
        cov = cov * annualize

    # 保证 PSD
    cov = (cov + cov.T) / 2.0
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += np.eye(len(assets)) * (-eigvals.min() + 1e-8)
    return mu_s, cov, assets
