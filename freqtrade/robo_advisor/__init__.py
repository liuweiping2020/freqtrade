"""
Freqtrade Robo-Advisor & Intelligent Investment Research Module.

基于 Freqtrade 的智能投顾与智能投研扩展模块，提供：
    - 用户画像与风险评估 (user_profile)
    - 资产配置引擎 (asset_allocator)
    - 因子研究全流程 (factor_purifier, factor_analyzer, factor_tools)
    - 金融 NLP 能力 (nlp)
    - 多智能体投研框架 (agents)
    - 投资组合再平衡 (portfolio_rebalancer)
    - 绩效报告与收益归因 (report_engine)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # P0 用户 & 资产配置
    "InvestorProfile",
    "RiskQuestionnaire",
    "RiskAssessor",
    "BaseAllocator",
    "AllocationResult",
    "MarkowitzAllocator",
    "RiskParityAllocator",
    "BlackLittermanAllocator",
    # P1 因子研究
    "FactorPurifier",
    "FactorAnalyzer",
    "FactorCombiner",
    "BarraAttributor",
    # P2 NLP
    "SentimentAnalyzer",
    "ReportParser",
    # P3 多智能体
    "BaseAgent",
    "Orchestrator",
    "ResearchContext",
    "InvestmentDecision",
    "AgentOutput",
    "FundamentalAnalyst",
    "TechnicalsAnalyst",
    "SentimentAnalyst",
    "RiskController",
    "PortfolioManager",
    # P3 组合再平衡
    "PortfolioRebalancer",
    "RebalanceOrder",
    "RebalanceInfo",
    # P4 报告
    "ReportEngine",
    "PerformanceSummary",
    "BrinsonResult",
    "RiskDecomposition",
]


def __getattr__(name):
    """Lazy import to reduce startup overhead."""
    _import_map = {
        # P0
    "InvestorProfile": (".user_profile", "InvestorProfile"),
    "RiskQuestionnaire": (".user_profile", "RiskQuestionnaire"),
    "RiskAssessor": (".user_profile", "RiskAssessor"),
    "BaseAllocator": (".asset_allocator", "BaseAllocator"),
    "AllocationResult": (".asset_allocator", "AllocationResult"),
    "MarkowitzAllocator": (".asset_allocator", "MarkowitzAllocator"),
    "RiskParityAllocator": (".asset_allocator", "RiskParityAllocator"),
    "BlackLittermanAllocator": (".asset_allocator", "BlackLittermanAllocator"),
        # P1
        "FactorPurifier": (".factor_purifier", "FactorPurifier"),
        "FactorAnalyzer": (".factor_analyzer", "FactorAnalyzer"),
        "FactorCombiner": (".factor_tools", "FactorCombiner"),
        "BarraAttributor": (".factor_tools", "BarraAttributor"),
        # P2
        "SentimentAnalyzer": (".nlp.sentiment", "SentimentAnalyzer"),
        "ReportParser": (".nlp.report_parser", "ReportParser"),
        # P3 核心
        "BaseAgent": (".agents.base", "BaseAgent"),
        "Orchestrator": (".agents.base", "Orchestrator"),
        "ResearchContext": (".agents.base", "ResearchContext"),
        "InvestmentDecision": (".agents.base", "InvestmentDecision"),
        "AgentOutput": (".agents.base", "AgentOutput"),
        # P3 职能智能体
        "FundamentalAnalyst": (".agents.analysts", "FundamentalAnalyst"),
        "TechnicalsAnalyst": (".agents.analysts", "TechnicalsAnalyst"),
        "SentimentAnalyst": (".agents.analysts", "SentimentAnalyst"),
        "RiskController": (".agents.analysts", "RiskController"),
        "PortfolioManager": (".agents.analysts", "PortfolioManager"),
        # P3 再平衡
        "PortfolioRebalancer": (".portfolio_rebalancer", "PortfolioRebalancer"),
        "RebalanceOrder": (".portfolio_rebalancer", "RebalanceOrder"),
        "RebalanceInfo": (".portfolio_rebalancer", "RebalanceInfo"),
        # P4 报告
        "ReportEngine": (".report_engine", "ReportEngine"),
        "PerformanceSummary": (".report_engine", "PerformanceSummary"),
        "BrinsonResult": (".report_engine", "BrinsonResult"),
        "RiskDecomposition": (".report_engine", "RiskDecomposition"),
    }
    if name not in _import_map:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, cls_name = _import_map[name]
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, cls_name)
