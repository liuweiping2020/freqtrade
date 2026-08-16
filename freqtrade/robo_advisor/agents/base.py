"""
多智能体投研框架核心 (agents/base)
=====================================

参考 TradingAgents / AI Hedge Fund 的多智能体协作模式，实现一个轻量、
无外部 LLM 依赖也能跑通、同时可接入真实大模型的投研编排框架。

核心抽象：

    BaseAgent        投研智能体基类（基本面分析师 / 技术分析师 / 舆情分析师 …）
      ├─ name/role/description
      ├─ analyze(context: ResearchContext) -> AgentOutput
      └─ revise(prev_outputs, context) -> AgentOutput   # 多轮辩论用

    AgentOutput      统一输出：
                       - direction: +1 / 0 / -1 (看多/中性/空)
                       - confidence: 0~1
                       - per_asset_scores: {asset: -1..1}
                       - reasoning: 论点字符串列表
                       - metadata: 原始数据（供下游）

    Orchestrator     编排器：
                       register(agent)
                       run(context, mode="sequential|majority|debate") -> InvestmentDecision

不依赖任何 LLM API：所有 BaseAgent 默认实现基于规则（例如「均线金叉 = 看多」）；
如需 LLM 只要继承 BaseAgent 并覆盖 analyze()，内部自行调用 OpenAI/Kronos/Claude 即可。
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 枚举/数据结构
# ---------------------------------------------------------------------------


class Signal(IntEnum):
    SHORT = -1
    NEUTRAL = 0
    LONG = 1


@dataclass
class ResearchContext:
    """
    投研上下文（所有 agent 共享的全局输入）。

    典型内容：
        symbols:       研究的资产列表
        date:          当前投研锚点日期
        market:        市场环境标签（由上游识别：trend/range/high_vol/crash）
        prices:        OHLCV 数据 {symbol: DataFrame}，可选
        factor_scores: 因子研究结果 {factor_name: {symbol: score}}，可选
        news:          近期新闻 [{symbol, title, sentiment, source, published_at}]
        profile:       InvestorProfile（来自 user_profile，用于约束仓位）
        allocation:    上次推荐的权重 {asset: w}
        extras:        任意扩展数据
    """

    symbols: list[str]
    date: Any = None  # pandas.Timestamp / date
    market_regime: str = "range"
    prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    factor_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    news: list[dict[str, Any]] = field(default_factory=list)
    profile: Any = None  # InvestorProfile
    allocation: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])


@dataclass
class AgentOutput:
    """智能体统一输出结构。"""

    agent_name: str
    direction: Signal = Signal.NEUTRAL
    confidence: float = 0.5                         # 0 ~ 1
    per_asset_scores: dict[str, float] = field(default_factory=dict)  # symbol ∈ [-1, 1]
    reasoning: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # --- 工具 ------------------------------------------------------------

    def clip_scores(self) -> "AgentOutput":
        """强制 per_asset_scores ∈ [-1, 1]"""
        self.per_asset_scores = {
            k: float(np.clip(v, -1.0, 1.0)) for k, v in self.per_asset_scores.items()
        }
        if self.confidence > 1.0:
            self.confidence = 1.0
        if self.confidence < 0.0:
            self.confidence = 0.0
        return self


@dataclass
class InvestmentDecision:
    """编排器最终输出：投资决策。"""

    request_id: str
    mode: str
    # 组合级
    overall_direction: Signal = Signal.NEUTRAL       # 组合总体多空倾向
    aggregate_confidence: float = 0.5
    # 资产级
    target_weights: dict[str, float] = field(default_factory=dict)   # symbol -> weight (sum≈1)
    per_asset_scores: dict[str, float] = field(default_factory=dict)  # symbol ∈ [-1,1]
    # 文本级
    summary: str = ""
    agent_summaries: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    # 原始
    agent_outputs: list[AgentOutput] = field(default_factory=list)
    context: ResearchContext | None = None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 智能体基类
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    投研智能体基类。

    最少实现 analyze(context) -> AgentOutput 即可。
    若要参与 debate（多轮修改），可选覆盖 revise()。
    """

    name: str = "base_agent"
    role: str = "analyst"
    description: str = "基础分析师"

    def __init__(self, llm_client: Any = None, **config: Any) -> None:
        """
        llm_client: 可选，任何兼容 `generate(prompt)->str` 的对象，
        用于覆盖 analyze/reaon 中的规则逻辑，接入真实大模型。
        """
        self.llm_client = llm_client
        self.config = config

    # ---- 主入口 ---------------------------------------------------------

    @abstractmethod
    def analyze(self, context: ResearchContext) -> AgentOutput:
        ...

    def revise(
        self,
        context: ResearchContext,
        own_prev: AgentOutput,
        peer_outputs: list[AgentOutput],
    ) -> AgentOutput:
        """
        辩论阶段：参考同行的结论修订自己的输出。
        默认实现：无修改。子类可覆盖为「加权平均」或 LLM 审阅。
        """
        return own_prev

    # ---- LLM 辅助 ------------------------------------------------------

    def _llm_generate(self, prompt: str, **kwargs) -> str | None:
        if self.llm_client is None:
            return None
        try:
            if hasattr(self.llm_client, "generate"):
                return str(self.llm_client.generate(prompt, **kwargs))
            if callable(self.llm_client):
                return str(self.llm_client(prompt, **kwargs))
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# 编排器
# ---------------------------------------------------------------------------


OrchestrationMode = Literal["sequential", "majority", "debate"]


class Orchestrator:
    """
    多智能体编排器。

    支持三种协作模式：
      - sequential:  逐个智能体依次分析，后续可读取前序产出（通过 context.extras['agent_outputs']）
      - majority:    所有智能体并行分析，然后对方向/分数进行加权投票
      - debate:      第一轮 analyze → 智能体互相查看结论 revise() → 第二轮投票；
                     可通过 debate_rounds=N 迭代 N 轮

    Usage::

        orch = Orchestrator(mode="majority", vote_by_confidence=True)
        orch.register(FundamentalAnalyst())
        orch.register(TechnicalsAnalyst())
        orch.register(SentimentAnalyst())
        decision = orch.run(ctx)
        print(decision.summary, decision.target_weights)
    """

    def __init__(
        self,
        mode: OrchestrationMode = "majority",
        vote_by_confidence: bool = True,
        debate_rounds: int = 2,
        weight_bounds: tuple[float, float] = (0.0, 0.4),
        long_only: bool = True,
    ) -> None:
        self.mode = mode
        self.vote_by_confidence = vote_by_confidence
        self.debate_rounds = debate_rounds
        self.w_lo, self.w_hi = weight_bounds
        self.long_only = long_only
        self.agents: list[BaseAgent] = []
        # 每个 agent 的投票权重（默认为 1 / N，可自定义）
        self._agent_weights: dict[str, float] = {}

    # ---- 注册 -----------------------------------------------------------

    def register(self, agent: BaseAgent, weight: float = 1.0) -> "Orchestrator":
        if any(a.name == agent.name for a in self.agents):
            raise ValueError(f"Agent name {agent.name!r} already registered")
        self.agents.append(agent)
        self._agent_weights[agent.name] = weight
        return self

    def set_agent_weight(self, agent_name: str, weight: float) -> None:
        if agent_name not in self._agent_weights:
            raise KeyError(f"Unknown agent: {agent_name}")
        self._agent_weights[agent_name] = weight

    # ---- 主入口 ---------------------------------------------------------

    def run(self, context: ResearchContext) -> InvestmentDecision:
        if not self.agents:
            raise RuntimeError("No agents registered. Use Orchestrator.register() first.")
        if self.mode == "sequential":
            outputs = self._run_sequential(context)
        elif self.mode == "majority":
            outputs = self._run_parallel(context)
        elif self.mode == "debate":
            outputs = self._run_debate(context)
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")
        return self._synthesize(context, outputs)

    # ---- 每种模式的实现 ------------------------------------------------

    def _run_parallel(self, context: ResearchContext) -> list[AgentOutput]:
        return [self._safe_analyze(a, context) for a in self.agents]

    def _run_sequential(self, context: ResearchContext) -> list[AgentOutput]:
        outputs: list[AgentOutput] = []
        # 前序 outputs 挂到 context.extras，供下游 agent 参考
        ctx_extras_snapshot = dict(context.extras)
        try:
            for agent in self.agents:
                context.extras["agent_outputs"] = list(outputs)
                out = self._safe_analyze(agent, context)
                outputs.append(out)
        finally:
            context.extras.clear()
            context.extras.update(ctx_extras_snapshot)
        return outputs

    def _run_debate(self, context: ResearchContext) -> list[AgentOutput]:
        # 第一轮
        outputs = self._run_parallel(context)
        # N-1 轮修订
        for _ in range(max(self.debate_rounds - 1, 0)):
            next_outputs: list[AgentOutput] = []
            for agent, prev in zip(self.agents, outputs):
                others = [o for o in outputs if o.agent_name != prev.agent_name]
                try:
                    revised = agent.revise(context, prev, others)
                    revised.clip_scores()
                except Exception:
                    revised = prev
                next_outputs.append(revised)
            outputs = next_outputs
        return outputs

    @staticmethod
    def _safe_analyze(agent: BaseAgent, ctx: ResearchContext) -> AgentOutput:
        try:
            out = agent.analyze(ctx)
            out.clip_scores()
        except Exception as exc:
            out = AgentOutput(
                agent_name=agent.name,
                reasoning=[f"[ERROR] {type(exc).__name__}: {exc}"],
                metadata={"error": str(exc)},
            )
        return out

    # ---- 综合决策 ------------------------------------------------------

    def _synthesize(
        self, context: ResearchContext, outputs: list[AgentOutput]
    ) -> InvestmentDecision:
        # 1) 聚合资产级得分：加权（agent 权重 × 置信度）平均
        symbols = list(context.symbols)
        agg_scores: dict[str, list[tuple[float, float]]] = {s: [] for s in symbols}
        for out in outputs:
            aw = self._agent_weights.get(out.agent_name, 1.0)
            cw = out.confidence if self.vote_by_confidence else 1.0
            w_eff = max(aw * cw, 1e-6)
            for s in symbols:
                sc = out.per_asset_scores.get(s, 0.0)
                agg_scores[s].append((sc, w_eff))

        final_scores: dict[str, float] = {}
        for s, items in agg_scores.items():
            if not items:
                final_scores[s] = 0.0
                continue
            scs, wts = zip(*items)
            scs_arr = np.asarray(scs, dtype=float)
            wts_arr = np.asarray(wts, dtype=float)
            sw = wts_arr.sum()
            final_scores[s] = float((scs_arr * wts_arr).sum() / sw) if sw > 0 else 0.0

        # 2) 组合级方向：sgn(mean(final_scores * confidence_weights))
        mean_s = float(np.mean(list(final_scores.values()))) if final_scores else 0.0
        if mean_s > 0.05:
            overall = Signal.LONG
        elif mean_s < -0.05:
            overall = Signal.SHORT
        else:
            overall = Signal.NEUTRAL

        # 3) 置信度：跨 agent 的加权一致性
        confidences = [
            (out.confidence * self._agent_weights.get(out.agent_name, 1.0)) for out in outputs
        ]
        agg_conf = float(np.mean(confidences) / max(sum(self._agent_weights.get(o.agent_name, 1.0) for o in outputs), 1e-6))
        agg_conf = float(np.clip(agg_conf * len(outputs), 0.05, 0.98))  # 简单尺度校正

        # 4) 目标权重：正分数映射为做多权重（long_only → 负的置 0）
        weights = self._scores_to_weights(final_scores)

        # 5) 总结
        summaries: list[str] = []
        for out in outputs:
            direction = {Signal.LONG: "看多", Signal.NEUTRAL: "中性", Signal.SHORT: "看空"}[out.direction]
            head = f"[{out.agent_name}] {direction} (置信={out.confidence:.0%})"
            body = "; ".join(out.reasoning[:3]) if out.reasoning else "(无详细论点)"
            summaries.append(f"{head}: {body}")

        direction_cn = {Signal.LONG: "偏多", Signal.NEUTRAL: "中性震荡", Signal.SHORT: "偏空"}[overall]
        summary = (
            f"组合{direction_cn}（共识分={mean_s:+.2f}，置信={agg_conf:.0%}）；"
            f"Top3多头: {self._topn(weights, n=3, reverse=True)}；"
            f"Top3空头/低配: {self._topn(weights, n=3, reverse=False)}。"
        )

        risks = self._collect_risks(context, outputs, weights)

        return InvestmentDecision(
            request_id=context.request_id,
            mode=self.mode,
            overall_direction=overall,
            aggregate_confidence=agg_conf,
            target_weights=weights,
            per_asset_scores=final_scores,
            summary=summary,
            agent_summaries=summaries,
            risks=risks,
            agent_outputs=outputs,
            context=context,
        )

    # ---- 辅助 ---------------------------------------------------------

    def _scores_to_weights(self, scores: dict[str, float]) -> dict[str, float]:
        """
        [-1,1] 分数 → 多权重。
        策略：对分数做 softplus 变换 → clip 到 [w_lo, w_hi] → 归一到和=1。
        """
        if not scores:
            return {}
        keys = list(scores.keys())
        vals = np.array([scores[k] for k in keys], dtype=float)
        if self.long_only:
            vals = np.maximum(vals, 0.0)
        else:
            # 允许做空：|val| 代表仓位强度，正负代表方向
            pass
        # softplus 平滑 → 正值 → clip
        sp = np.logaddexp(vals, 0.0)  # softplus(x) = log(1+e^x) ∈ (0, +∞)
        sp = np.clip(sp, self.w_lo * 0.5 + 1e-9, self.w_hi)
        s = sp.sum()
        if s < 1e-9:
            # 等权 fallback
            sp = np.full_like(sp, 1.0 / len(sp))
        else:
            sp = sp / s
        return {k: round(float(w), 6) for k, w in zip(keys, sp)}

    @staticmethod
    def _topn(weights: dict[str, float], n: int = 3, reverse: bool = True) -> str:
        if not weights:
            return "（空）"
        items = sorted(weights.items(), key=lambda kv: kv[1], reverse=reverse)
        return ", ".join(f"{k}:{v:+.1%}" if v < 0 else f"{k}:{v:.1%}" for k, v in items[:n])

    @staticmethod
    def _collect_risks(ctx: ResearchContext, outs: Sequence[AgentOutput], w: dict[str, float]) -> list[str]:
        risks: list[str] = []
        # 集中度风险
        if w:
            max_w = max(w.values())
            if max_w > 0.35:
                risks.append(f"单资产集中度过高：max_weight={max_w:.1%}")
            hhi = sum(v * v for v in w.values())
            if hhi > 0.25:  # 等效于 4 个等权资产
                risks.append(f"权重集中度 HHI={hhi:.3f} 偏高，建议分散")
        # 观点分歧风险
        directions = [o.direction for o in outs]
        if Signal.LONG in directions and Signal.SHORT in directions:
            risks.append("分析师多空观点存在明显分歧，建议复核")
        # 市场 regime 提示
        if ctx.market_regime == "high_vol":
            risks.append("当前市场处于高波动状态，建议降低杠杆与单仓")
        if ctx.market_regime == "crash":
            risks.append("市场风险-off，建议现金比例提升并启动保护性止损")
        return risks
