"""
Freqtrade Robo-Advisor · REST API 服务入口
================================================

启动服务::

    python -m freqtrade.robo_advisor.api_server

访问地址（默认）：
    服务主页 & 文档  http://127.0.0.1:8765/
    OpenAPI JSON    http://127.0.0.1:8765/openapi.json
    Swagger UI      http://127.0.0.1:8765/docs
    Redoc           http://127.0.0.1:8765/redoc

REST Endpoints 一览：
    GET  /                                        首页（系统状态 & 入口链接）
    GET  /healthz                                 健康检查
    GET  /api/v1/questionnaire                    获取标准风险问卷问题
    POST /api/v1/profile/evaluate                 问卷答案 → InvestorProfile
    POST /api/v1/allocate/risk_parity             风险平价资产配置
    POST /api/v1/allocate/markowitz               Markowitz 均值-方差配置
    POST /api/v1/analyze/factor                   单因子诊断（IC / 分层回测）
    POST /api/v1/agents/decision                  多智能体协作决策
    POST /api/v1/nlp/sentiment                    批量情感分析
    POST /api/v1/nlp/report_parse                 研报结构化抽取
    POST /api/v1/rebalance/check                  当前权重偏离检测 + 订单建议
    POST /api/v1/report/summarize                 净值曲线 → 绩效报告（Markdown/JSON）
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from freqtrade.robo_advisor import (
    # 画像
    RiskQuestionnaire,
    RiskAssessor,
    InvestorProfile,
    # 资产配置
    RiskParityAllocator,
    MarkowitzAllocator,
    AllocationResult,
    # 因子
    FactorPurifier,
    FactorAnalyzer,
    # 编排
    ResearchContext,
    Orchestrator,
    FundamentalAnalyst,
    TechnicalsAnalyst,
    RiskController,
    PortfolioManager,
    # NLP
    SentimentAnalyzer,
    ReportParser,
    # 再平衡
    PortfolioRebalancer,
    # 报告
    ReportEngine,
)


# ---------------------------------------------------------------------------
# 路由：请求 & 响应 Schema
# ---------------------------------------------------------------------------


# --- 问卷 / 画像
class EvaluateProfileRequest(BaseModel):
    answers: list[int] | None = Field(default=None, description="10 题风险问卷答案，每题 1~5；为空时根据 seed 自动生成示例")
    investment_horizon: Literal["short", "medium", "long"] = "medium"
    financial_goal: Literal["capital_preservation", "steady_growth", "aggressive_growth", "speculation"] = "steady_growth"
    seed: int | None = Field(default=None, description="若 answers 为空，按 seed 随机生成示例答案")


# --- 资产配置
class AllocateRequest(BaseModel):
    symbols: list[str] = Field(description="资产代号列表")
    returns_pct: dict[str, list[float]] = Field(description="每个资产的历史日收益率序列（小数，例如 0.01 = 1%）")
    min_weight: float = 0.0
    max_weight: float = 0.60
    group_constraints: list[dict[str, Any]] = Field(default_factory=list, description='例:[{"assets":["BTC","ETH"],"min":0.3,"max":0.8}]')
    # Markowitz 专用
    objective: Literal["max_sharpe", "min_volatility", "target_return", "target_volatility", "max_return"] = "max_sharpe"
    target_return: float | None = None
    target_volatility: float | None = None
    risk_free_rate: float = 0.04


# --- 因子分析
class AnalyzeFactorRequest(BaseModel):
    dates: list[str]
    assets: list[str]
    factor_values: list[list[float]]  # shape (n_dates, n_assets)
    forward_returns: list[list[float]]  # shape (n_dates, n_assets)
    quantiles: int = 2


# --- 多智能体决策
class AgentDecisionRequest(BaseModel):
    symbols: list[str]
    prices_by_symbol: dict[str, list[float]] = Field(description="每个 symbol 的最近收盘价序列（>=20 个点）")
    factor_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    profile_risk_level: Literal["R1", "R2", "R3", "R4", "R5"] = "R3"
    mode: Literal["sequential", "majority", "debate"] = "sequential"
    market_regime: Literal["trend", "range", "high_vol", "crash"] = "range"


# --- NLP
class SentimentRequest(BaseModel):
    texts: list[str]
    backend: Literal["lexicon", "transformers", "llm"] = "lexicon"


class ReportParseRequest(BaseModel):
    title: str = "未命名研报"
    text: str
    max_keywords: int = 30
    abstract_sentences: int = 3


# --- 再平衡
class RebalanceCheckRequest(BaseModel):
    current_weights: dict[str, float]
    target_weights: dict[str, float]
    current_prices: dict[str, float]
    total_portfolio_value_usdt: float
    last_rebalance_days_ago: int = 45
    threshold_individual: float = 0.05
    threshold_portfolio: float = 0.10
    schedule_days: int = 30
    trading_cost_bps: float = 15.0
    mode: Literal["threshold", "schedule", "threshold+schedule", "driftsync"] = "threshold+schedule"


# --- 报告
class ReportRequest(BaseModel):
    equity: dict[str, float]  # date_string -> portfolio value
    risk_free_rate: float = 0.04
    periods_per_year: float = 365.0
    output_format: Literal["json", "markdown"] = "markdown"


# --- Freqtrade 引擎
class BacktestRequest(BaseModel):
    strategy: str = Field(description="策略类名，如 SampleStrategy")
    exchange: str = Field(default="binance", description="交易所名称")
    pairs: list[str] = Field(default_factory=lambda: ["BTC/USDT"], description="回测品种列表")
    timeframe: str = Field(default="1h", description="K线周期")
    timerange: str = Field(default="20240101-20240601", description="回测时间范围")
    stake_currency: str | None = None
    stake_amount: str | None = None
    dry_run_wallet: float | None = None
    max_open_trades: int | None = None


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------


APP_TITLE = "Freqtrade Robo-Advisor · 智能投研 API"
APP_DESCRIPTION = """
基于 Freqtrade 的智能投研系统。提供：
风险问卷 & 画像、资产配置（风险平价 / Markowitz）、因子诊断、
多智能体投研决策、NLP 情感分析 & 研报抽取、组合再平衡、绩效报告 & 归因。
"""


def create_app() -> FastAPI:
    app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version="0.1.0")

    # 静态 Dashboard 目录：本文件同级 ui/
    UI_DIR = Path(__file__).resolve().parent / "ui"

    # 复用引擎实例（无状态，可全局）
    assessor = RiskAssessor()
    sentiment_lex = SentimentAnalyzer(backend="lexicon")
    report_parser = ReportParser()

    # -------- 首页 & 健康 --------
    @app.get("/", tags=["system"])
    async def index(request: Request):
        accept = request.headers.get("accept", "")
        prefers_html = "text/html" in accept and "application/json" not in accept
        if prefers_html:
            return RedirectResponse(url="/ui", status_code=302)
        return {
            "service": APP_TITLE,
            "status": "running",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dashboard_ui": "/ui",
            "endpoints": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_json": "/openapi.json",
                "healthz": "/healthz",
                "questionnaire": "/api/v1/questionnaire",
                "profile_evaluate": "/api/v1/profile/evaluate (POST)",
                "allocate_risk_parity": "/api/v1/allocate/risk_parity (POST)",
                "allocate_markowitz": "/api/v1/allocate/markowitz (POST)",
                "factor_analyze": "/api/v1/analyze/factor (POST)",
                "agent_decision": "/api/v1/agents/decision (POST)",
                "nlp_sentiment": "/api/v1/nlp/sentiment (POST)",
                "nlp_report_parse": "/api/v1/nlp/report_parse (POST)",
                "rebalance_check": "/api/v1/rebalance/check (POST)",
                "report_summarize": "/api/v1/report/summarize (POST)",
                "freqtrade_strategies": "/api/v1/freqtrade/strategies",
                "freqtrade_exchanges": "/api/v1/freqtrade/exchanges",
                "freqtrade_pairlists": "/api/v1/freqtrade/pairlists",
                "freqtrade_freqai_models": "/api/v1/freqtrade/freqai_models",
                "freqtrade_hyperopt_losses": "/api/v1/freqtrade/hyperopt_losses",
                "freqtrade_protections": "/api/v1/freqtrade/protections",
                "freqtrade_pair_history": "/api/v1/freqtrade/pair_history",
                "freqtrade_data_files": "/api/v1/freqtrade/data_files",
                "freqtrade_backtest": "/api/v1/freqtrade/backtest (POST)",
            },
        }

    @app.get("/ui", tags=["ui"], include_in_schema=False)
    @app.get("/ui/", tags=["ui"], include_in_schema=False)
    async def dashboard_ui():
        path = UI_DIR / "dashboard.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dashboard file not found: ui/dashboard.html")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @app.get("/healthz", tags=["system"])
    async def healthz():
        return {"status": "ok", "ts": time.time()}

    # -------- 问卷 & 画像 --------
    @app.get("/api/v1/questionnaire", tags=["profile"])
    async def get_questionnaire():
        return {
            "count": len(RiskQuestionnaire.QUESTIONS),
            "answers_range": "[1..5]",
            "weights": RiskQuestionnaire.WEIGHTS,
            "questions": [
                {"id": i + 1, "text": q}
                for i, q in enumerate(RiskQuestionnaire.QUESTIONS)
            ],
            "sample_generator": "POST /api/v1/profile/evaluate 时传 seed 可生成示例答案",
        }

    @app.post("/api/v1/profile/evaluate", tags=["profile"])
    async def evaluate_profile(req: EvaluateProfileRequest):
        answers = req.answers
        if not answers or len(answers) != 10:
            if req.seed is None:
                raise HTTPException(status_code=400, detail="answers 必须是长度 10 的列表（1~5），或提供 seed 生成示例")
            answers = RiskQuestionnaire.sample_answers(seed=req.seed)
        profile = assessor.evaluate(
            answers,
            investment_horizon=req.investment_horizon,
            financial_goal=req.financial_goal,
        )
        profile = profile.derive_constraints(override=True)
        mix = profile.suggest_asset_mix()
        return {
            "risk_score": profile.risk_score,
            "risk_level": profile.risk_level,
            "horizon": profile.investment_horizon,
            "financial_goal": profile.financial_goal,
            "max_drawdown_tolerance": profile.max_drawdown_tolerance,
            "max_leverage": profile.suggested_max_leverage,
            "suggest_asset_mix": mix,
            "freqtrade_constraints": profile.to_freqtrade_constraints(),
        }

    # -------- 资产配置：风险平价 --------
    def _build_returns_df(syms, ret_pct):
        if set(ret_pct.keys()) != set(syms):
            raise HTTPException(status_code=400, detail=f"returns_pct 中的资产必须与 symbols 完全一致，缺失 {set(syms)-set(ret_pct.keys())}")
        lengths = {len(v) for v in ret_pct.values()}
        if len(lengths) > 1:
            raise HTTPException(status_code=400, detail="所有资产的收益率序列长度必须相同")
        return pd.DataFrame({s: ret_pct[s] for s in syms})

    @app.post("/api/v1/allocate/risk_parity", tags=["allocate"])
    async def allocate_rp(req: AllocateRequest):
        rets_df = _build_returns_df(req.symbols, req.returns_pct)
        cov = (rets_df.cov() * 252).to_numpy()
        alloc = RiskParityAllocator(default_max_weight=req.max_weight, default_min_weight=req.min_weight)
        res: AllocationResult = alloc.allocate(
            assets=req.symbols, cov_matrix=cov, expected_returns=None,
            group_constraints=req.group_constraints or None,
        )
        return _serialize_allocation(res)

    @app.post("/api/v1/allocate/markowitz", tags=["allocate"])
    async def allocate_mv(req: AllocateRequest):
        rets_df = _build_returns_df(req.symbols, req.returns_pct)
        cov = (rets_df.cov() * 252).to_numpy()
        mu = (rets_df.mean() * 252).to_numpy()
        alloc = MarkowitzAllocator(objective=req.objective, default_max_weight=req.max_weight, default_min_weight=req.min_weight)
        res: AllocationResult = alloc.allocate(
            assets=req.symbols, cov_matrix=cov, expected_returns=mu,
            group_constraints=req.group_constraints or None,
            risk_free_rate=req.risk_free_rate,
            target_return=req.target_return, target_volatility=req.target_volatility,
        )
        return _serialize_allocation(res)

    # -------- 因子分析 --------
    @app.post("/api/v1/analyze/factor", tags=["factor"])
    async def analyze_factor(req: AnalyzeFactorRequest):
        if len(req.dates) != len(req.factor_values) or len(req.dates) != len(req.forward_returns):
            raise HTTPException(status_code=400, detail="dates / factor_values / forward_returns 第 0 维（日期）必须一致")
        if len(req.assets) != len(req.factor_values[0]) or len(req.assets) != len(req.forward_returns[0]):
            raise HTTPException(status_code=400, detail="assets 长度必须与 factor/forward 的第 1 维一致")
        dates = pd.to_datetime(req.dates)
        # 堆叠成 MultiIndex [date, asset]
        idx = pd.MultiIndex.from_product([dates, req.assets], names=["date", "asset"])
        f_vals = np.asarray(req.factor_values, dtype=float).ravel()
        r_vals = np.asarray(req.forward_returns, dtype=float).ravel()
        factor_series = pd.Series(f_vals, index=idx, name="factor")
        forward_df = pd.DataFrame({"ret_1d": r_vals}, index=idx)
        analyzer = FactorAnalyzer(quantiles=req.quantiles, min_assets_per_date=max(2, len(req.assets)))
        result = analyzer.run(factor_series, forward_df, factor_name="factor")
        ic_s = result.ic_summary()
        qp = result.quantile_performance().reset_index().to_dict(orient="records")
        return {
            "n_dates": len(req.dates),
            "n_assets": len(req.assets),
            "ic_summary": {k: (float(v) if pd.notna(v) else None) for k, v in ic_s.items()},
            "quantile_performance": qp,
        }

    # -------- 多智能体 --------
    @app.post("/api/v1/agents/decision", tags=["agents"])
    async def agent_decision(req: AgentDecisionRequest):
        # 1) InvestorProfile（从 risk_level 直接构造一个简化）
        rl = req.profile_risk_level
        risk_score_map = {"R1": 2, "R2": 4, "R3": 6, "R4": 8, "R5": 10}
        profile = InvestorProfile(
            risk_score=risk_score_map[rl],
            investment_horizon="long",
            financial_goal="steady_growth",
        ).derive_constraints(override=True)
        # 2) 价格 → {symbol: DataFrame(OHLCV)}
        prices_dict: dict[str, pd.DataFrame] = {}
        for s in req.symbols:
            if s not in req.prices_by_symbol:
                raise HTTPException(400, detail=f"缺少 {s} 的价格序列")
            close = np.asarray(req.prices_by_symbol[s], dtype=float)
            if len(close) < 20:
                raise HTTPException(400, detail=f"{s} 价格点不足 20 个")
            dates = pd.date_range(end=pd.Timestamp.now("UTC").normalize(), periods=len(close), freq="D")
            # 用 close 近似合成 OHLCV（避免调用端必须给全）
            rng = np.random.default_rng(abs(hash(s)) & 0xFFFFFFFF)
            noise = np.abs(rng.normal(0, 0.003, size=len(close)))
            sdf = pd.DataFrame({
                "open": close * (1 - noise * 0.5),
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "volume": rng.integers(1000, 100000, size=len(close)).astype(float),
            }, index=dates)
            prices_dict[s] = sdf
        # 3) 构造 context
        ctx = ResearchContext(
            symbols=req.symbols,
            date=pd.Timestamp.now("UTC").normalize(),
            market_regime=req.market_regime,
            prices=prices_dict,
            factor_scores=req.factor_scores or {},
            profile=profile,
        )
        # 4) 编排
        orch = Orchestrator(mode=req.mode)
        orch.register(FundamentalAnalyst())
        orch.register(TechnicalsAnalyst())
        orch.register(RiskController())
        orch.register(PortfolioManager())
        decision = orch.run(ctx)
        ranked = sorted(decision.per_asset_scores.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "request_id": decision.request_id,
            "mode": decision.mode,
            "market_regime": req.market_regime,
            "overall_direction": decision.overall_direction.name,
            "aggregate_confidence": float(decision.aggregate_confidence) if not np.isnan(decision.aggregate_confidence) else None,
            "per_asset_scores": decision.per_asset_scores,
            "target_weights": {k: round(v, 6) for k, v in decision.target_weights.items()},
            "ranked_symbols": [s for s, _ in ranked],
            "summary": decision.summary,
            "risks": decision.risks,
        }

    # -------- NLP --------
    @app.post("/api/v1/nlp/sentiment", tags=["nlp"])
    async def nlp_sentiment(req: SentimentRequest):
        backend_map = {"lexicon": "lexicon", "transformers": "auto", "llm": "llm_client"}
        sa = SentimentAnalyzer(backend=backend_map[req.backend])
        results = sa.analyze_batch(req.texts)
        return {
            "backend": req.backend,
            "count": len(results),
            "avg_score": float(np.mean([r.score for r in results])) if results else 0.0,
            "results": [
                {
                    "text": r.text,
                    "label": r.label,
                    "score": r.score,
                    "model": r.model,
                }
                for r in results
            ],
        }

    @app.post("/api/v1/nlp/report_parse", tags=["nlp"])
    async def nlp_report_parse(req: ReportParseRequest):
        res = report_parser.parse(
            req.text, title=req.title,
            max_keywords=req.max_keywords,
            abstract_sentences=req.abstract_sentences,
        )
        return res.to_dict()

    # -------- 再平衡 --------
    @app.post("/api/v1/rebalance/check", tags=["rebalance"])
    async def rebalance_check(req: RebalanceCheckRequest):
        rb = PortfolioRebalancer(
            threshold_individual=req.threshold_individual,
            threshold_portfolio=req.threshold_portfolio,
            schedule_days=req.schedule_days,
            trading_cost_bps=req.trading_cost_bps,
            mode=req.mode,
        )
        last_ts = time.time() - req.last_rebalance_days_ago * 86400
        orders, info = rb.run(
            current_weights=req.current_weights,
            target_weights=req.target_weights,
            current_prices=req.current_prices,
            total_portfolio_value_usdt=req.total_portfolio_value_usdt,
            last_rebalance_ts=last_ts,
        )
        return {
            "triggered": info.triggered,
            "reasons": info.trigger_reasons,
            "individual_max_deviation": info.individual_max_deviation,
            "portfolio_deviation": info.portfolio_deviation,
            "days_since_last_rebalance": info.days_since_last_rebalance,
            "estimated_cost_usdt": info.estimated_trading_cost_usdt,
            "estimated_gain_usdt": info.estimated_gain_from_rebalance,
            "prior_weights": info.prior_weights,
            "post_weights": info.post_weights,
            "orders": [
                {
                    "symbol": o.symbol,
                    "side": o.side.name,
                    "target_weight": round(o.target_weight, 6),
                    "current_weight": round(o.current_weight, 6),
                    "weight_delta": round(o.weight_delta, 6),
                    "notional_usdt": round(o.notional_usdt, 4),
                    "base_qty": round(o.base_qty, 8),
                    "reason": o.reason,
                }
                for o in orders
            ],
        }

    # -------- 报告 --------
    @app.post("/api/v1/report/summarize", tags=["report"])
    async def report_summarize(req: ReportRequest):
        eq = pd.Series(
            {pd.to_datetime(k): float(v) for k, v in req.equity.items()}
        ).sort_index()
        engine = ReportEngine(periods_per_year=req.periods_per_year, risk_free_rate=req.risk_free_rate)
        s = engine.summarize(eq)
        if req.output_format == "markdown":
            return {"format": "markdown", "content": engine.to_markdown(s, title="组合绩效报告")}
        # JSON
        mdd = s.max_drawdown_info
        json_obj = {
            "format": "json",
            "total_return": float(s.total_return),
            "annual_return": float(s.annual_return),
            "annual_volatility": float(s.annual_volatility),
            "sharpe": float(s.sharpe_ratio),
            "sortino": float(s.sortino_ratio),
            "calmar": float(s.calmar_ratio),
            "max_drawdown": float(s.max_drawdown),
            "mdd_info": {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in mdd.items()},
            "var_95": float(s.var_95),
            "cvar_95": float(s.cvar_95),
            "win_rate": float(s.win_rate),
            "profit_factor": float(s.profit_factor),
            "meta": s.meta,
            "monthly_returns_pct": (s.monthly_returns * 100).round(2).fillna(-1).to_dict(),
        }
        return json_obj

    # ================================================================
    # Freqtrade 引擎能力
    # ================================================================
    WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
    FREQTRADE_PKG = Path(__file__).resolve().parents[1]
    USER_DATA_DIR = WORKSPACE_ROOT / "user_data"

    # --- 支持的交易所（硬编码 fallback，避免链式 import 失败）---
    _SUPPORTED_EXCHANGES = [
        "binance", "binanceus", "binanceusdm", "bingx", "bitmart", "bitget",
        "bitpanda", "bitvavo", "bybit", "bybiteu", "coinex", "cryptocom",
        "gate", "gateeu", "hitbtc", "htx", "hyperliquid", "idex",
        "kraken", "krakenfutures", "kucoin", "lbank", "luno", "modetrade",
        "myokx", "okx", "okxus",
    ]

    def _scan_py_classes(directory: Path, base_prefix: str) -> list[dict]:
        """扫描目录下 *.py，用正则提取 class 名（不做 import，避免重依赖）。"""
        results: list[dict] = []
        if not directory.exists():
            return results
        for f in sorted(directory.glob("*.py")):
            if f.name.startswith("__"):
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            for m in re.finditer(r"^class\s+(\w+)\s*\(([^)]+)\)\s*:", text, re.MULTILINE):
                cls_name = m.group(1)
                bases = m.group(2).strip()
                # 跳过接口/基类
                if cls_name.startswith("I") and cls_name in ("IPairList", "IHyperOptLoss", "IResolver"):
                    continue
                # 提取 docstring 第一行
                doc = ""
                doc_m = re.search(r'"""(.*?)"""', text[m.end():m.end() + 500], re.DOTALL)
                if doc_m:
                    doc = doc_m.group(1).strip().split("\n")[0].strip()
                results.append({
                    "name": cls_name,
                    "file": f.name,
                    "bases": bases,
                    "doc": doc[:120],
                })
        return results

    @app.get("/api/v1/freqtrade/strategies", tags=["freqtrade"])
    async def ft_strategies():
        """枚举 user_data/strategies/ 下所有策略文件。"""
        strat_dir = USER_DATA_DIR / "strategies"
        items: list[dict] = []
        # 同时扫描 user_data/strategies 和 freqtrade/templates
        for directory in [strat_dir, FREQTRADE_PKG / "templates"]:
            if not directory.exists():
                continue
            for f in sorted(directory.glob("*.py")):
                if f.name.startswith("__") or f.name == "sample_hyperopt_loss.py":
                    continue
                try:
                    text = f.read_text(encoding="utf-8")
                except Exception:
                    continue
                for m in re.finditer(r"^class\s+(\w+)\s*\((?:IStrategy|.*IStrategy.*)\)\s*:", text, re.MULTILINE):
                    doc = ""
                    doc_m = re.search(r'"""(.*?)"""', text[m.end():m.end() + 500], re.DOTALL)
                    if doc_m:
                        doc = doc_m.group(1).strip().split("\n")[0].strip()
                    items.append({
                        "name": m.group(1),
                        "file": f.name,
                        "path": str(f),
                        "source": "user_data" if "user_data" in str(f) else "template",
                        "doc": doc[:120],
                    })
        return {"count": len(items), "strategies": items}

    @app.get("/api/v1/freqtrade/exchanges", tags=["freqtrade"])
    async def ft_exchanges():
        """支持的交易所列表。"""
        # 尝试动态 import，失败则用硬编码
        exchanges = _SUPPORTED_EXCHANGES
        try:
            from freqtrade.exchange.common import SUPPORTED_EXCHANGES  # type: ignore
            exchanges = list(SUPPORTED_EXCHANGES)
        except Exception:
            pass
        return {"count": len(exchanges), "exchanges": sorted(exchanges)}

    @app.get("/api/v1/freqtrade/pairlists", tags=["freqtrade"])
    async def ft_pairlists():
        """可用的 PairList 过滤器/生成器清单。"""
        items = _scan_py_classes(FREQTRADE_PKG / "plugins" / "pairlist", "pairlist")
        return {"count": len(items), "pairlists": items}

    @app.get("/api/v1/freqtrade/freqai_models", tags=["freqtrade"])
    async def ft_freqai_models():
        """可用的 FreqAI 预测模型清单。"""
        items = _scan_py_classes(FREQTRADE_PKG / "freqai" / "prediction_models", "freqai")
        return {"count": len(items), "models": items}

    @app.get("/api/v1/freqtrade/hyperopt_losses", tags=["freqtrade"])
    async def ft_hyperopt_losses():
        """可用的 Hyperopt 损失函数清单。"""
        items = _scan_py_classes(FREQTRADE_PKG / "optimize" / "hyperopt_loss", "hyperopt")
        # 损失函数文件名带 hyperopt_loss_ 前缀，取类名更友好
        for item in items:
            item["short_name"] = item["name"].replace("HyperOptLoss", "").replace("HyperoptLoss", "")
        return {"count": len(items), "losses": items}

    @app.get("/api/v1/freqtrade/protections", tags=["freqtrade"])
    async def ft_protections():
        """可用的保护策略清单。"""
        items = _scan_py_classes(FREQTRADE_PKG / "plugins" / "protections", "protection")
        return {"count": len(items), "protections": items}

    @app.get("/api/v1/freqtrade/pair_history", tags=["freqtrade"])
    async def ft_pair_history(
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
        timerange: str | None = None,
        candle_type: str = "spot",
    ):
        """加载本地历史 OHLCV 数据（从 user_data/data/ 读取）。"""
        from freqtrade.data.history import load_pair_history
        from freqtrade.enums.candletype import CandleType
        from freqtrade.configuration import TimeRange

        data_dir = USER_DATA_DIR / "data"
        tr = TimeRange.parse_timerange(timerange) if timerange else None
        try:
            df = load_pair_history(
                pair=pair,
                timeframe=timeframe,
                datadir=data_dir,
                timerange=tr,
                candle_type=CandleType.from_string(candle_type),
            )
        except Exception as e:
            raise HTTPException(400, detail=f"加载失败: {e}")
        if df is None or df.empty:
            raise HTTPException(404, detail=f"未找到 {pair} {timeframe} 数据；请先下载（POST /api/v1/freqtrade/download_data）")
        # 返回最近 500 根 + 统计
        tail = df.tail(500)
        return {
            "pair": pair,
            "timeframe": timeframe,
            "candle_type": candle_type,
            "total_rows": len(df),
            "returned_rows": len(tail),
            "first_date": str(df.index[0]) if len(df) else None,
            "last_date": str(df.index[-1]) if len(df) else None,
            "columns": list(df.columns),
            "ohlcv": [
                {
                    "date": str(idx),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
                for idx, row in tail.iterrows()
            ],
        }

    @app.get("/api/v1/freqtrade/data_files", tags=["freqtrade"])
    async def ft_data_files():
        """列出 user_data/data/ 下已有的数据文件（按交易所分目录）。"""
        data_root = USER_DATA_DIR / "data"
        result: dict[str, list[str]] = {}
        if not data_root.exists():
            return {"data_dir": str(data_root), "exchanges": {}}
        for ex_dir in sorted(data_root.iterdir()):
            if ex_dir.is_dir():
                files = sorted([f.name for f in ex_dir.iterdir() if f.is_file()])
                result[ex_dir.name] = files[:50]  # 限制数量
        return {"data_dir": str(data_root), "exchanges": result}

    @app.post("/api/v1/freqtrade/backtest", tags=["freqtrade"])
    async def ft_backtest(req: BacktestRequest):
        """运行回测（需要本地已有历史数据）。"""
        try:
            from freqtrade.configuration import Configuration
            from freqtrade.optimize.backtesting import Backtesting
        except ImportError as e:
            raise HTTPException(503, detail=f"Freqtrade 回测模块不可用（缺少依赖）: {e}")

        # 构造最小 config
        config = {
            "user_data_dir": str(USER_DATA_DIR),
            "strategy": req.strategy,
            "timeframe": req.timeframe,
            "timerange": req.timerange,
            "datadir": str(USER_DATA_DIR / "data" / req.exchange),
            "exchange": {"name": req.exchange, "key": "", "secret": "", "pair_whitelist": req.pairs, "pair_blacklist": []},
            "stake_currency": req.stake_currency or "USDT",
            "stake_amount": req.stake_amount or "unlimited",
            "dry_run_wallet": req.dry_run_wallet or 1000,
            "max_open_trades": req.max_open_trades or 3,
            "trading_mode": "spot",
            "margin_mode": "",
        }
        # 加载策略验证
        try:
            bt = Backtesting(config)
            bt.start()
        except Exception as e:
            raise HTTPException(400, detail=f"回测失败: {e}")

        # 从最后一份回测结果提取摘要
        stats = getattr(bt, "results", None)
        if not stats:
            return {"status": "completed", "detail": "回测完成但未提取到统计摘要，请查看 user_data/backtest_results/"}

        # 尝试提取关键指标
        strategy_stats = stats.get("strategy", {})
        result: dict[str, Any] = {"status": "completed", "strategies": list(strategy_stats.keys())}
        for strat_name, s in strategy_stats.items():
            result[strat_name] = {
                "total_trades": s.get("total_trades", 0),
                "profit_total": s.get("profit_total", 0),
                "profit_total_abs": s.get("profit_total_abs", 0),
                "max_drawdown": s.get("max_drawdown", 0),
                "max_drawdown_abs": s.get("max_drawdown_abs", 0),
                "sharpe": s.get("sharpe", 0),
                "sortino": s.get("sortino", 0),
                "calmar": s.get("calmar", 0),
                "winrate": s.get("winrate", 0),
                "avg_trade_duration": s.get("holding_avg", ""),
                "best_pair": s.get("best_pair", ""),
                "worst_pair": s.get("worst_pair", ""),
                "market_change": s.get("market_change", 0),
            }
        return result

    return app


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _serialize_allocation(res: AllocationResult) -> dict:
    weights = {res.assets[i]: float(res.weights[i]) for i in range(len(res.assets))}
    rc = {res.assets[i]: float(res.risk_contributions[i]) for i in range(len(res.assets))}
    return {
        "model": res.model,
        "weights": weights,
        "expected_return": float(res.expected_return),
        "volatility": float(res.volatility),
        "sharpe": float(res.sharpe),
        "diversify_ratio": float(res.diversify_ratio),
        "risk_contributions": rc,
        "turnover": float(res.turnover),
    }


app = create_app()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ROBO_API_PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
