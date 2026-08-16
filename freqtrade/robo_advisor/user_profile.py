"""
用户画像与风险评估模块 (user_profile)
======================================

功能：
    1. InvestorProfile       - 投资者画像数据结构
    2. RiskQuestionnaire     - 风险承受能力问卷（10 题标准版）
    3. RiskAssessor          - 问卷打分 → 画像生成 → 投资约束映射

设计理念（参考 KYC / 适当性管理）：
    - 风险等级 1~10 对应 R1~R5 五级分类
    - 根据画像推导：max_drawdown_tolerance、suggested_leverage、asset_mix_hint
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# 枚举 & 字面量类型
# ---------------------------------------------------------------------------


class RiskLevel(IntEnum):
    """R1(保守) ~ R5(激进) 五级风险分类"""

    R1 = 1  # 保守型 - 本金安全优先，可接受极低波动
    R2 = 2  # 稳健型 - 小幅波动，追求稳健增值
    R3 = 3  # 平衡型 - 风险收益平衡
    R4 = 4  # 进取型 - 可承受较大波动，追求较高收益
    R5 = 5  # 激进型 - 可承受大幅波动/回撤，最大化长期收益


InvestmentHorizon = Literal["short", "medium", "long"]
# short:  < 1 年   | 流动性优先，货币/短债类为主
# medium: 1~3 年  | 平衡配置
# long:   > 3 年  | 权益类可以更高


FinancialGoal = Literal["capital_preservation", "steady_growth", "aggressive_growth", "speculation"]
IncomeLevel = Literal["low", "lower_middle", "middle", "upper_middle", "high"]
InvestmentExperience = Literal["none", "beginner", "intermediate", "advanced", "expert"]
MarketRegime = Literal["trend", "range", "high_vol", "crash"]


# ---------------------------------------------------------------------------
# 投资者画像
# ---------------------------------------------------------------------------


class InvestorProfile(BaseModel):
    """
    投资者画像 - 驱动资产配置与策略选择的核心输入。

    Fields:
        risk_score (int):               1~10 综合风险得分（由问卷得到）
        risk_level (RiskLevel):         R1~R5，由 risk_score 映射
        investment_horizon:             投资期限
        financial_goal:                 理财目标
        income_level:                   月收入档位
        investment_experience:          投资经验
        liquid_net_worth (float):       可投资净资产（元/USDT），用于头寸校准
        annual_income_stability (float): 0~1，收入稳定性系数（1=极度稳定）
        max_drawdown_tolerance (float): 可容忍最大回撤（-0.05 = -5%），可由画像推导或手动覆盖
        suggested_max_leverage (float): 建议最大杠杆（1=无杠杆）
        preferred_asset_classes:        偏好资产类别（crypto/equity/bond/gold/cash）
        constraints:                    自定义硬性约束（黑名单、单资产上限等）
    """

    # ---------- 基本属性 ----------
    risk_score: int = Field(default=5, ge=1, le=10)
    risk_level: RiskLevel = RiskLevel.R3
    investment_horizon: InvestmentHorizon = "medium"
    financial_goal: FinancialGoal = "steady_growth"
    income_level: IncomeLevel = "middle"
    investment_experience: InvestmentExperience = "beginner"

    # ---------- 财务属性 ----------
    liquid_net_worth: float = Field(default=10_000.0, gt=0)
    annual_income_stability: float = Field(default=0.6, ge=0, le=1)

    # ---------- 风险约束（可手动覆盖，否则由 derive_constraints 推导）----------
    max_drawdown_tolerance: float = Field(default=-0.15, le=0, ge=-1.0)
    suggested_max_leverage: float = Field(default=1.0, ge=0.5, le=10.0)
    preferred_asset_classes: list[str] = Field(
        default_factory=lambda: ["crypto", "cash"]
    )
    constraints: dict[str, Any] = Field(default_factory=dict)

    # ---------- 元数据 ----------
    profile_id: str = "default"
    created_at: float = Field(default_factory=lambda: __import__("time").time())
    updated_at: float = Field(default_factory=lambda: __import__("time").time())

    model_config = {"validate_assignment": True, "extra": "allow"}

    @field_validator("preferred_asset_classes")
    @classmethod
    def _validate_asset_classes(cls, v: list[str]) -> list[str]:
        allowed = {"crypto", "equity", "bond", "gold", "cash", "commodity", "forex"}
        if not v:
            raise ValueError("preferred_asset_classes cannot be empty")
        for item in v:
            if item not in allowed:
                raise ValueError(f"Unknown asset class {item!r}, allowed: {allowed}")
        return v

    @model_validator(mode="after")
    def _sync_risk_level(self) -> "InvestorProfile":
        """risk_score ↔ risk_level 自动同步"""
        # 基于 risk_score → R1~R5
        mapping = [(2, RiskLevel.R1), (4, RiskLevel.R2), (6, RiskLevel.R3), (8, RiskLevel.R4), (10, RiskLevel.R5)]
        for upper, level in mapping:
            if self.risk_score <= upper:
                self.risk_level = level
                break
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def derive_constraints(self, override: bool = False) -> "InvestorProfile":
        """
        根据画像自动推导 `max_drawdown_tolerance` 与 `suggested_max_leverage`。
        仅当字段仍为默认值或 override=True 时生效。
        """
        # 风险等级 → 最大回撤容忍（绝对值越大越激进）
        dd_map: dict[RiskLevel, float] = {
            RiskLevel.R1: -0.03,
            RiskLevel.R2: -0.07,
            RiskLevel.R3: -0.15,
            RiskLevel.R4: -0.28,
            RiskLevel.R5: -0.50,
        }
        # 风险等级 → 建议杠杆
        lev_map: dict[RiskLevel, float] = {
            RiskLevel.R1: 1.0,
            RiskLevel.R2: 1.0,
            RiskLevel.R3: 1.5,
            RiskLevel.R4: 2.5,
            RiskLevel.R5: 4.0,
        }
        # 经验 → 杠杆修正（新手降杠杆）
        exp_factor: dict[InvestmentExperience, float] = {
            "none": 0.5,
            "beginner": 0.75,
            "intermediate": 1.0,
            "advanced": 1.2,
            "expert": 1.4,
        }
        # 期限 → 回撤容忍修正（长期可容忍更大波动）
        horizon_factor: dict[InvestmentHorizon, float] = {
            "short": 0.6,
            "medium": 1.0,
            "long": 1.3,
        }
        defaults_dd = -0.15
        defaults_lev = 1.0

        if override or math.isclose(self.max_drawdown_tolerance, defaults_dd, abs_tol=1e-6):
            self.max_drawdown_tolerance = dd_map[self.risk_level] * horizon_factor[self.investment_horizon]

        if override or math.isclose(self.suggested_max_leverage, defaults_lev, abs_tol=1e-6):
            self.suggested_max_leverage = round(
                lev_map[self.risk_level] * exp_factor[self.investment_experience], 2
            )

        self.updated_at = __import__("time").time()
        return self

    def suggest_asset_mix(self) -> dict[str, float]:
        """
        返回建议的大类资产权重基准（用于资产配置引擎的先验 / 权重约束）。

        Returns:
            dict: {"crypto": 0.3, "equity": 0.2, "bond": 0.2, "gold": 0.05, "cash": 0.25, ...}
        """
        # R1 → 现金+债券为主；R5 → 权益+加密为主
        level = self.risk_level
        base_mix = {
            RiskLevel.R1: {"crypto": 0.02, "equity": 0.05, "bond": 0.50, "gold": 0.03, "cash": 0.40},
            RiskLevel.R2: {"crypto": 0.08, "equity": 0.15, "bond": 0.40, "gold": 0.05, "cash": 0.32},
            RiskLevel.R3: {"crypto": 0.20, "equity": 0.25, "bond": 0.28, "gold": 0.07, "cash": 0.20},
            RiskLevel.R4: {"crypto": 0.38, "equity": 0.30, "bond": 0.15, "gold": 0.07, "cash": 0.10},
            RiskLevel.R5: {"crypto": 0.55, "equity": 0.25, "bond": 0.06, "gold": 0.06, "cash": 0.08},
        }[level]
        # 期限因子：长期更多权益
        horizon_boost = {"short": 0.7, "medium": 1.0, "long": 1.2}[self.investment_horizon]
        for k in ("crypto", "equity"):
            base_mix[k] = float(np.clip(base_mix[k] * horizon_boost, 0.0, 0.9))
        # 归一到 1
        s = sum(base_mix.values())
        return {k: round(v / s, 4) for k, v in base_mix.items() if k in self.preferred_asset_classes}

    # ------------------------------------------------------------------
    # Freqtrade 对接辅助
    # ------------------------------------------------------------------

    def to_freqtrade_constraints(self) -> dict[str, Any]:
        """
        将画像约束转为 Freqtrade 可用的 config patch。

        示例：
            {
                "max_open_trades": 3,
                "stake_amount": "unlimited" / 100.0,
                "stoploss": -0.05,
                "trading_mode": "spot" / "futures",
                "leverage": 1.0,
            }
        """
        rl = self.risk_level
        max_trades_map = {RiskLevel.R1: 2, RiskLevel.R2: 3, RiskLevel.R3: 5, RiskLevel.R4: 7, RiskLevel.R5: 10}
        # 建议单笔 stake_amount 占可投资资产的比例
        stake_pct_map = {RiskLevel.R1: 0.05, RiskLevel.R2: 0.08, RiskLevel.R3: 0.12, RiskLevel.R4: 0.15, RiskLevel.R5: 0.20}
        return {
            "max_open_trades": max_trades_map[rl],
            "stake_amount": round(self.liquid_net_worth * stake_pct_map[rl], 2),
            "stoploss": round(self.max_drawdown_tolerance / max_trades_map[rl], 4),  # 单交易止损 ≈ 总回撤 / 持仓数
            "trading_mode": "futures" if self.suggested_max_leverage > 1.01 else "spot",
            "leverage": self.suggested_max_leverage,
        }


# ---------------------------------------------------------------------------
# 风险评估问卷
# ---------------------------------------------------------------------------


@dataclass
class RiskQuestionnaire:
    """
    标准 10 题风险承受能力问卷（参考基金投顾 KYC 问卷设计）。

    每题 1~5 分，总分 10~50 分，归一到 risk_score 1~10。

    问题覆盖五个维度：
        - 投资经验（题 1~2）
        - 投资期限（题 3）
        - 亏损承受（题 4~6）
        - 波动容忍（题 7~8）
        - 财务状况（题 9~10）
    """

    answers: list[int] = field(default_factory=lambda: [3] * 10)

    # 类属性（可被子类/外部直接访问与覆盖）
    QUESTIONS: list[str] = [
        # 维度：投资经验
        "Q1. 您的投资年限？(1:<1年 2:1-3年 3:3-5年 4:5-10年 5:>10年)",
        "Q2. 您主要接触过哪些投资品类？(1:存款/理财 2:基金 3:股票 4:期货/期权 5:加密资产等另类)",
        # 维度：期限
        "Q3. 这笔资金计划投资多久？(1:<6月 2:6-12月 3:1-3年 4:3-5年 5:>5年)",
        # 维度：亏损承受
        "Q4. 若投资 10 万元，您可接受最多亏损多少不焦虑？(1:<3千 2:5千 3:1.5万 4:3万 5:>5万)",
        "Q5. 投资亏损 20% 后，您的行为是？(1:立即赎回 2:减仓 3:持有不动 4:小幅加仓 5:大幅加仓)",
        "Q6. 您的本金承受底线是？(1:绝不能亏损 2:最多亏3% 3:最多亏10% 4:最多亏25% 5:50%以上可接受)",
        # 维度：波动容忍
        "Q7. 您偏好的收益特征是？(1:稳稳正收益 2:小幅波动 3:波动适中 4:较大波动 5:高波动高收益)",
        "Q8. 单月最大浮亏多少您能接受？(1:<2% 2:5% 3:10% 4:20% 5:>30%)",
        # 维度：财务状况
        "Q9. 可投资资产占您家庭总资产的比例？(1:<5% 2:10% 3:25% 4:50% 5:>70%)",
        "Q10. 您的家庭月结余率（结余/收入）？(1:<10% 2:20% 3:40% 4:60% 5:>80%)",
    ]

    # 各题在综合风险分中的权重（可按业务调节，和 QUESTIONS 一一对应，和为 1.0）
    WEIGHTS: list[float] = [
        0.10, 0.10,  # 经验
        0.12,        # 期限
        0.14, 0.12, 0.12,  # 亏损
        0.10, 0.10,        # 波动
        0.05, 0.05,        # 财务
    ]

    def __post_init__(self) -> None:
        if len(self.answers) != 10:
            raise ValueError("RiskQuestionnaire requires exactly 10 answers")
        for i, a in enumerate(self.answers):
            if not (1 <= a <= 5):
                raise ValueError(f"Answer {i} must be in [1..5], got {a}")

    def raw_score(self) -> float:
        """加权原始得分 ∈ [1, 5]"""
        return float(np.average(self.answers, weights=self.WEIGHTS))

    def to_risk_score(self) -> int:
        """归一到 [1, 10] 的整数 risk_score"""
        rs = self.raw_score()  # 1..5
        return int(np.clip(round((rs - 1) / 4 * 9 + 1), 1, 10))


# ---------------------------------------------------------------------------
# 风险评估引擎
# ---------------------------------------------------------------------------


class RiskAssessor:
    """
    风险评估引擎：问卷 + 补充输入 → InvestorProfile。

    典型用法::

        assessor = RiskAssessor()
        answers = [3, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        profile = assessor.evaluate(
            questionnaire_answers=answers,
            investment_horizon="medium",
            liquid_net_worth=50000.0,
        )
        print(profile.risk_level, profile.suggest_asset_mix())
    """

    def __init__(self, custom_weights: list[float] | None = None) -> None:
        self._custom_weights = custom_weights

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        questionnaire_answers: list[int],
        investment_horizon: InvestmentHorizon = "medium",
        financial_goal: FinancialGoal = "steady_growth",
        income_level: IncomeLevel = "middle",
        investment_experience: InvestmentExperience = "beginner",
        liquid_net_worth: float = 10_000.0,
        annual_income_stability: float = 0.6,
        preferred_asset_classes: list[str] | None = None,
        profile_id: str = "default",
    ) -> InvestorProfile:
        """执行评估并返回 InvestorProfile（已自动推导约束）。"""
        q = RiskQuestionnaire(answers=questionnaire_answers)
        if self._custom_weights is not None:
            q.WEIGHTS = self._custom_weights  # type: ignore[attr-defined]
        risk_score = q.to_risk_score()

        # 经验自动校准（根据 Q1 Q2 答案，如果用户没显式覆盖的话）
        if investment_experience == "beginner":
            exp_avg = (questionnaire_answers[0] + questionnaire_answers[1]) / 2
            exp_map = [
                (1.4, "none"),
                (2.2, "beginner"),
                (3.0, "intermediate"),
                (4.0, "advanced"),
                (5.1, "expert"),
            ]
            for upper, label in exp_map:
                if exp_avg <= upper:
                    investment_experience = label  # type: ignore[assignment]
                    break

        profile = InvestorProfile(
            risk_score=risk_score,
            investment_horizon=investment_horizon,
            financial_goal=financial_goal,
            income_level=income_level,
            investment_experience=investment_experience,
            liquid_net_worth=liquid_net_worth,
            annual_income_stability=annual_income_stability,
            preferred_asset_classes=preferred_asset_classes or ["crypto", "cash"],
            profile_id=profile_id,
        )
        # 用画像本身的启发式推导约束
        return profile.derive_constraints(override=True)

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    @staticmethod
    def list_questions() -> list[str]:
        """返回问卷题目列表（便于前端渲染）"""
        return RiskQuestionnaire.QUESTIONS  # type: ignore[return-value]
