"""
配置文件 Schema (examples/config_schemas)
==========================================

给整个 robo_advisor 体系定义「配置文件」格式（Pydantic）。
通过 YAML / JSON 加载，再创建各模块实例即可，推荐用于生产：

    with open("robo_advisor.yaml", "r", encoding="utf-8") as f:
        cfg = RoboAdvisorConfig.model_validate(yaml.safe_load(f))
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --- 顶层枚举 ------------------------------------------------------------


class AllocatorType(str, Enum):
    MARKOWITZ = "markowitz"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"


class PurifierPreset(str, Enum):
    DEFAULT = "default"
    STRICT = "strict"
    NEURAL = "neural"


class OrchestratorMode(str, Enum):
    SEQUENTIAL = "sequential"
    MAJORITY = "majority"
    DEBATE = "debate"


class RebalanceMode(str, Enum):
    THRESHOLD = "threshold"
    SCHEDULE = "schedule"
    THRESHOLD_SCHEDULE = "threshold+schedule"
    DRIFTSYNC = "driftsync"


# --- 各模块配置 ----------------------------------------------------------


class UserProfileConfig(BaseModel):
    questionnaire_variant: Literal["standard", "simple"] = "standard"
    default_risk_level: Literal["R1", "R2", "R3", "R4", "R5"] = "R3"
    default_horizon_months: int = 24
    default_goal: str = "wealth_preservation"
    override: dict[str, Any] = Field(default_factory=dict)


class AllocatorConfig(BaseModel):
    type: AllocatorType = AllocatorType.RISK_PARITY
    # Markowitz 专用
    objective: Literal["max_sharpe", "min_volatility", "target_return", "target_volatility", "max_return"] = "max_sharpe"
    target_return: float | None = None   # 年化（非周期）
    target_volatility: float | None = None
    min_weight: float = 0.0
    max_weight: float = 0.60
    group_constraints: dict[str, dict[str, float]] = Field(default_factory=dict)   # {"crypto": {"min":0.3, "max":0.7}}
    turnover_limit: float | None = None  # 单期最大换手（一次买卖和），如 0.40 = 40%
    risk_free_rate: float = 0.04


class FactorConfig(BaseModel):
    purifier_preset: PurifierPreset = PurifierPreset.DEFAULT
    custom_steps: list[str] = Field(default_factory=list)    # 如 ["HandleMissing","Winsorize","Neutralize","Standardize"]
    combine_method: Literal["equal_weight", "icir_weight", "symmetric_orth", "ltr_weighted"] = "icir_weight"
    ic_lookback: int = 60              # IC 计算窗口
    quantile_groups: int = 5


class AgentsConfig(BaseModel):
    mode: OrchestratorMode = OrchestratorMode.SEQUENTIAL
    enable: dict[str, bool] = Field(default_factory=lambda: {
        "fundamental": True, "technicals": True, "sentiment": True,
        "risk": True, "portfolio": True,
    })
    risk_checks: dict[str, float] = Field(default_factory=lambda: {
        "max_concentration": 0.6,
        "min_score_bear_regime": -0.5,
    })
    max_revision_rounds: int = 2


class NLPSentimentConfig(BaseModel):
    preferred_backend: Literal["lexicon", "transformers", "llm"] = "lexicon"
    transformers_model: str = "ProsusAI/finbert"
    batch_size: int = 32
    # LLM 配置：可选
    llm_model: str | None = None
    llm_temperature: float = 0.0


class NLPParserConfig(BaseModel):
    max_keywords: int = 50
    abstract_sentences: int = 3
    use_llm_summary: bool = False


class NLPConfig(BaseModel):
    sentiment: NLPSentimentConfig = Field(default_factory=NLPSentimentConfig)
    parser: NLPParserConfig = Field(default_factory=NLPParserConfig)


class RebalanceConfig(BaseModel):
    mode: RebalanceMode = RebalanceMode.THRESHOLD_SCHEDULE
    threshold_individual: float = 0.05
    threshold_portfolio: float = 0.10
    schedule_days: int = 30
    trading_cost_bps: float = 15.0
    min_notional_usdt: float = 20.0
    cash_buffer_weight: float = 0.0


class ReportConfig(BaseModel):
    periods_per_year: float = 365.0
    risk_free_rate: float = 0.04
    output_format: Literal["markdown", "html", "json", "pdf"] = "markdown"


# --- 顶层配置 ------------------------------------------------------------


class UniverseSymbol(BaseModel):
    symbol: str
    category: str = "crypto"        # crypto / equity / bond / commodity / cash
    benchmark_weight: float | None = None
    black_listed: bool = False


class RoboAdvisorConfig(BaseModel):
    """智能投研系统的顶层配置。"""

    name: str = "Freqtrade-RoboAdvisor"
    universe: list[UniverseSymbol] = Field(default_factory=list)

    user_profile: UserProfileConfig = Field(default_factory=UserProfileConfig)
    allocator: AllocatorConfig = Field(default_factory=AllocatorConfig)
    factor: FactorConfig = Field(default_factory=FactorConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # 数据 / 交易接口
    freqtrade_strategy: str | None = None
    data_backend: Literal["freqtrade", "csv", "parquet", "tushare", "yfinance", "ccxt"] = "freqtrade"

    # 自定义扩展
    extra: dict[str, Any] = Field(default_factory=dict)


# --- YAML 示例 -----------------------------------------------------------


SAMPLE_YAML = """
name: "Demo-RoboAdvisor"

universe:
  - { symbol: BTC,  category: crypto, benchmark_weight: 0.50 }
  - { symbol: ETH,  category: crypto, benchmark_weight: 0.25 }
  - { symbol: SOL,  category: crypto, benchmark_weight: 0.10 }
  - { symbol: BNB,  category: crypto, benchmark_weight: 0.05 }
  - { symbol: USDT, category: cash,   benchmark_weight: 0.10 }

user_profile:
  default_risk_level: R3
  default_horizon_months: 36
  default_goal: balanced_growth

allocator:
  type: risk_parity
  max_weight: 0.60
  turnover_limit: 0.40
  risk_free_rate: 0.04

factor:
  purifier_preset: default
  combine_method: icir_weight
  ic_lookback: 60
  quantile_groups: 5

agents:
  mode: sequential
  enable:
    fundamental: true
    technicals: true
    sentiment: true
    risk: true
    portfolio: true

nlp:
  sentiment:
    preferred_backend: lexicon
  parser:
    max_keywords: 50
    abstract_sentences: 3

rebalance:
  mode: "threshold+schedule"
  threshold_individual: 0.05
  threshold_portfolio: 0.10
  schedule_days: 30
  trading_cost_bps: 15
  min_notional_usdt: 20

report:
  periods_per_year: 365
  risk_free_rate: 0.04
  output_format: markdown

data_backend: csv
"""
