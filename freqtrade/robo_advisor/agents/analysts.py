"""
职能型投研智能体 (agents/analysts)
=====================================

实现 5 类典型投研岗位 Agent（默认规则引擎，可替换为 LLM 版）：

    FundamentalAnalyst  基本面分析师  → 估值/链上/宏观指标评分
    TechnicalsAnalyst   技术分析师     → MA、RSI、MACD、布林、ATR 等信号综合
    SentimentAnalyst    舆情分析师     → 新闻/社交媒体情绪分
    RiskController      风控分析师     → 波动率/VaR/回撤约束的风险调整打分
    PortfolioManager    组合经理       → 结合 InvestorProfile 约束的最终建议

每个 Agent 的 analyze() 返回标准 AgentOutput，per_asset_scores ∈ [-1,1]。
如提供 llm_client，可在 analyze() 中以自然语言生成更详细 reasoning；
默认不依赖 LLM，使用确定性规则即可完成端到端演示。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import (
    AgentOutput,
    BaseAgent,
    ResearchContext,
    Signal,
)


# ---------------------------------------------------------------------------
# 辅助：技术指标（避免依赖 talib，用 pandas/numpy 实现常见几个）
# ---------------------------------------------------------------------------


def _ma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=max(1, n // 2)).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = 2 * (dif - dea)
    return dif, dea, hist


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = _ma(close, n)
    std = close.rolling(n, min_periods=max(1, n // 2)).std(ddof=0)
    return mid - k * std, mid, mid + k * std


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


# ---------------------------------------------------------------------------
# 1) 基本面 / 链上分析
# ---------------------------------------------------------------------------


class FundamentalAnalyst(BaseAgent):
    """
    基本面/估值分析师。

    从 context.factor_scores 读取标准的估值/质量/链上因子列：
        - "valuation":   pe/pb/mvrv/nvt 类估值分（0~1 或 zscore，越大越高估）
        - "quality":     盈利/资产质量分，越大越优质
        - "momentum_6m": 中期动量，越大越强
        - "macro":       宏观流动性/利率/美元分，越大对风险资产越友好
    """

    name = "fundamental"
    role = "基本面分析师"
    description = "综合估值、质量、动量、宏观维度给出内在价值评分"

    WEIGHTS_DEFAULT = {"valuation": 0.30, "quality": 0.25, "momentum_6m": 0.25, "macro": 0.20}

    def __init__(self, weights: dict[str, float] | None = None, neutral_threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights or dict(self.WEIGHTS_DEFAULT)
        self.neutral_threshold = neutral_threshold

    def analyze(self, context: ResearchContext) -> AgentOutput:
        scores: dict[str, float] = {}
        reasons: list[str] = []

        # 确保归一化到 [-1, +1]：对每个因子映射
        def _norm(v: float, factor: str) -> float:
            # 估值 valuation：高 = 利空（映射为负）
            if factor == "valuation":
                # v ∈ [-2, +2] zscore → +2 贵 → 分数 -1
                return float(np.clip(-v / 2.0, -1.0, 1.0))
            # 其他：高 = 利多
            return float(np.clip(v / 2.0, -1.0, 1.0))

        for s in context.symbols:
            score = 0.0
            contrib = []
            for f, w in self.weights.items():
                v = context.factor_scores.get(f, {}).get(s, 0.0)
                contrib.append((f, v))
                score += _norm(v, f) * w
            scores[s] = float(np.clip(score, -1.0, 1.0))

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        if avg > self.neutral_threshold:
            direction = Signal.LONG
            reasons.append("估值合理偏低 + 质量改善 + 宏观友好")
        elif avg < -self.neutral_threshold:
            direction = Signal.SHORT
            reasons.append("估值偏高或质量恶化，建议谨慎")
        else:
            direction = Signal.NEUTRAL
            reasons.append("基本面处于中性区间，没有极端高估/低估")
        # Top/bottom 结论
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        bot = sorted(scores.items(), key=lambda x: x[1])[:3]
        reasons.append("基本面TOP3: " + ", ".join(f"{k}({v:+.2f})" for k, v in top))
        reasons.append("基本面BOT3: " + ", ".join(f"{k}({v:+.2f})" for k, v in bot))

        conf = float(np.clip(0.5 + abs(avg) * 0.45, 0.4, 0.95))
        return AgentOutput(
            agent_name=self.name,
            direction=direction,
            confidence=conf,
            per_asset_scores=scores,
            reasoning=reasons,
        )


# ---------------------------------------------------------------------------
# 2) 技术分析师
# ---------------------------------------------------------------------------


class TechnicalsAnalyst(BaseAgent):
    """
    技术分析师：读取 context.prices[symbol]（OHLCV DataFrame）。
    综合 MA 交叉、RSI、MACD 柱、布林位置、趋势强度，给出 [-1, 1] 打分。
    """

    name = "technicals"
    role = "技术分析师"
    description = "技术面多因子信号综合（MA/RSI/MACD/BOLL/ATR）"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_one(self, df: pd.DataFrame) -> tuple[float, list[str]]:
        notes = []
        c = df["close"] if "close" in df.columns else df.iloc[:, 0]
        h = df["high"] if "high" in df.columns else c
        l = df["low"] if "low" in df.columns else c
        total = 0.0
        # --- MA20 vs MA60 ---
        ma20, ma60 = _ma(c, 20), _ma(c, 60)
        if len(ma20) >= 1 and len(ma60) >= 1:
            diff_pct = (ma20.iloc[-1] / ma60.iloc[-1] - 1) * 100 if ma60.iloc[-1] != 0 else 0
            sc = np.clip(diff_pct / 5.0, -1.0, 1.0)
            total += 0.30 * sc
            notes.append(f"MA20/60={diff_pct:+.2f}%")
        # --- RSI ---
        r = _rsi(c, 14)
        if len(r) >= 1:
            rv = float(r.iloc[-1])
            # 超买超卖的反向：RSI 70+ → 负分；30- → 正分
            sc = -np.clip((rv - 50) / 25, -1.0, 1.0)
            total += 0.25 * sc
            notes.append(f"RSI={rv:.0f}")
        # --- MACD histogram sign & magnitude ---
        _, _, hist = _macd(c, 12, 26, 9)
        if len(hist) >= 1:
            hv = float(hist.iloc[-1]) / (float(c.iloc[-1]) + 1e-9) * 100  # % 价格尺度
            sc = np.clip(hv / 0.5, -1.0, 1.0)
            total += 0.25 * sc
            notes.append(f"MACD柱={hv:+.3f}%")
        # --- 布林带位置 ---
        lo, mid, hi = _bollinger(c, 20, 2.0)
        if len(lo) >= 1 and (hi.iloc[-1] - lo.iloc[-1]) > 0:
            pos = (float(c.iloc[-1]) - float(lo.iloc[-1])) / (float(hi.iloc[-1]) - float(lo.iloc[-1]))
            # 靠近下轨 → 正面（均值回归），靠近上轨 → 负面
            sc = -np.clip((pos - 0.5) * 2, -1.0, 1.0)
            total += 0.20 * sc
            notes.append(f"BOLL位置={pos:.0%}")
        return float(np.clip(total, -1.0, 1.0)), notes

    def analyze(self, context: ResearchContext) -> AgentOutput:
        scores: dict[str, float] = {}
        all_notes: list[str] = []
        for s in context.symbols:
            df = context.prices.get(s)
            if df is None or len(df) < 30:
                scores[s] = 0.0
                all_notes.append(f"{s}: 数据不足，中性")
                continue
            sc, notes = self._score_one(df)
            scores[s] = sc
            all_notes.append(f"{s}({sc:+.2f}): {'; '.join(notes)}")

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        if avg > 0.08:
            dir = Signal.LONG
        elif avg < -0.08:
            dir = Signal.SHORT
        else:
            dir = Signal.NEUTRAL
        conf = float(np.clip(0.5 + abs(avg) * 0.5, 0.4, 0.92))
        reasoning = [f"技术面整体评分={avg:+.2f}，信号方向={dir.name}"] + all_notes[:8]
        return AgentOutput(
            agent_name=self.name,
            direction=dir,
            confidence=conf,
            per_asset_scores=scores,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# 3) 舆情分析师
# ---------------------------------------------------------------------------


class SentimentAnalyst(BaseAgent):
    """
    舆情分析师：聚合 context.news 中每条新闻的情绪标签。

    news 单条结构约定::

        { "symbol": "BTC", "title": "...", "sentiment": +1/0/-1,
          "confidence": 0.8, "source": "reuters", "published_at": ts }
    """

    name = "sentiment"
    role = "舆情分析师"
    description = "新闻/社交媒体情绪聚合"

    def __init__(self, decay_half_life_days: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.hl_days = decay_half_life_days

    def analyze(self, context: ResearchContext) -> AgentOutput:
        scores: dict[str, float] = {s: 0.0 for s in context.symbols}
        counts: dict[str, float] = {s: 0.0 for s in context.symbols}
        now = context.created_at or time_now()
        per_sym_reason: dict[str, list[str]] = {s: [] for s in context.symbols}

        for item in context.news:
            sym = item.get("symbol")
            if sym not in scores:
                continue
            s = float(item.get("sentiment", 0.0))
            c = float(item.get("confidence", 0.5))
            # 时间衰减
            dt_days = max(0.0, (now - float(item.get("published_at", now))) / 86400.0)
            decay = 0.5 ** (dt_days / max(self.hl_days, 0.01))
            w = c * decay + 1e-6
            scores[sym] += s * w
            counts[sym] += w
            if abs(s) >= 0.5 and len(per_sym_reason[sym]) < 2:
                title = item.get("title", "")[:30]
                tag = "+" if s > 0 else ("-" if s < 0 else "~")
                per_sym_reason[sym].append(f"{tag}{title}")

        final_scores: dict[str, float] = {}
        for sym in scores:
            if counts[sym] > 0:
                val = scores[sym] / counts[sym]  # ∈ [-1, +1]
                # 样本量修正：新闻越少越向 0 收缩
                shrunk = val * (1 - np.exp(-counts[sym] / 3.0))
                final_scores[sym] = float(np.clip(shrunk, -1.0, 1.0))
            else:
                final_scores[sym] = 0.0

        avg = float(np.mean(list(final_scores.values()))) if final_scores else 0.0
        if avg > 0.05:
            dir = Signal.LONG
        elif avg < -0.05:
            dir = Signal.SHORT
        else:
            dir = Signal.NEUTRAL
        reasoning = [f"舆情总分={avg:+.2f}，综合情绪={dir.name}"]
        for s in context.symbols:
            tag_score = f"{s}: score={final_scores[s]:+.2f}"
            if per_sym_reason[s]:
                tag_score += f"；新闻样例：{'/'.join(per_sym_reason[s])}"
            reasoning.append(tag_score)
        conf = float(np.clip(0.5 + abs(avg) * 0.5 + min(counts[s] for s in scores) * 0.05, 0.4, 0.9))
        return AgentOutput(
            agent_name=self.name,
            direction=dir,
            confidence=conf,
            per_asset_scores=final_scores,
            reasoning=reasoning,
            metadata={"news_counts": counts},
        )


def time_now() -> float:
    import time
    return time.time()


# ---------------------------------------------------------------------------
# 4) 风控分析师（对已有打分做风险调整）
# ---------------------------------------------------------------------------


class RiskController(BaseAgent):
    """
    风控分析师：
        - 单资产波动率（过去 N 日）过高 → 向中性收缩
        - VaR/ES 超限 → 压低分数
        - 组合当前权重/画像的最大回撤容忍 → 全局调权因子
        - 高波动 / 崩盘中的 regime → 降低风险偏好
    """

    name = "risk_control"
    role = "风控分析"
    description = "波动率/回撤/VaR/画像约束下的风险偏好校准"

    def __init__(self, lookback: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback

    def analyze(self, context: ResearchContext) -> AgentOutput:
        vols: dict[str, float] = {}
        for s in context.symbols:
            df = context.prices.get(s)
            if df is None or len(df) < 10:
                vols[s] = 0.02  # 默认 2%
                continue
            c = df["close"] if "close" in df.columns else df.iloc[:, 0]
            ret = c.pct_change().dropna().tail(self.lookback)
            vols[s] = float(ret.std(ddof=0)) if len(ret) > 1 else 0.02

        # 全局风险衰减：市场 regime + 画像最大回撤容忍
        regime_mult = {"trend": 1.0, "range": 0.85, "high_vol": 0.55, "crash": 0.25}
        mult_regime = regime_mult.get(context.market_regime, 0.85)
        if context.profile is not None and hasattr(context.profile, "max_drawdown_tolerance"):
            # dd ∈ [-0.5, -0.03]，越保守（大负值）→ mult 越低
            dd = float(context.profile.max_drawdown_tolerance)  # 负
            # 线性映射：dd=-3% → 0.3，dd=-50% → 1.0
            mult_dd = float(np.clip(((-dd) - 0.03) / 0.47, 0.3, 1.0))
        else:
            mult_dd = 0.75
        global_mult = mult_regime * mult_dd

        # 单资产：波动率越高 → 越向中性收缩
        median_vol = float(np.median(list(vols.values()))) if vols else 0.02
        median_vol = max(median_vol, 1e-6)
        scores: dict[str, float] = {}
        for s, v in vols.items():
            shrink = 1.0 / (1.0 + v / median_vol)   # 高波动 → 强收缩
            # 风控自身的方向是「零」，它只做收缩（配合下游 orchestrator 加权投票后产生结果）。
            # 这里不主动给方向，给中性 ± 根据 regime 稍微偏向保守
            if context.market_regime in ("high_vol", "crash"):
                base = -0.15
            else:
                base = 0.0
            scores[s] = float(np.clip((base * shrink) * global_mult, -1.0, 1.0))

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        direction = Signal.NEUTRAL
        if avg < -0.05:
            direction = Signal.SHORT
        reasoning = [
            f"市场 regime={context.market_regime} 因子={mult_regime:.2f}；画像风险乘子={mult_dd:.2f}；全局缩放宽={global_mult:.2f}",
            f"截面年化波动率 median={median_vol*np.sqrt(365):.1%}，top 高波动："
            + ", ".join(f"{k}({v*np.sqrt(365):.0%})" for k, v in sorted(vols.items(), key=lambda x: x[1], reverse=True)[:3]),
        ]
        return AgentOutput(
            agent_name=self.name,
            direction=direction,
            confidence=0.85,  # 风控永远高置信
            per_asset_scores=scores,
            reasoning=reasoning,
            metadata={"volatilities": vols, "global_multiplier": global_mult},
        )


# ---------------------------------------------------------------------------
# 5) 组合经理（二次修正：权重边界 + 画像约束 + 风险预算）
# ---------------------------------------------------------------------------


class PortfolioManager(BaseAgent):
    """
    组合经理：
        - 阅读前序 agent 的输出（context.extras['agent_outputs']，sequential 模式下）
        - 结合 InvestorProfile 的硬约束，输出最终 per_asset_scores（通常在 orchestrator 后作为最终 checkpoint）
    """

    name = "portfolio_manager"
    role = "组合经理"
    description = "结合画像约束的资产分数二次校准"

    def analyze(self, context: ResearchContext) -> AgentOutput:
        prev = context.extras.get("agent_outputs") or []
        # 从前序 agent 的 per_asset_scores 中读取
        per: dict[str, list[tuple[float, float]]] = {s: [] for s in context.symbols}
        for p in prev:
            if not isinstance(p, AgentOutput):
                continue
            cw = p.confidence
            for s in context.symbols:
                sc = p.per_asset_scores.get(s, 0.0)
                per[s].append((sc, cw))

        scores: dict[str, float] = {}
        for s, items in per.items():
            if not items:
                scores[s] = 0.0
                continue
            scs, wts = zip(*items)
            scs_arr = np.asarray(scs, dtype=float)
            wts_arr = np.asarray(wts, dtype=float) + 1e-6
            scores[s] = float(np.clip((scs_arr * wts_arr).sum() / wts_arr.sum(), -1.0, 1.0))

        # 画像约束：R1~R2 压低整体分数幅度；R4~R5 允许放大
        if context.profile is not None:
            rl_num = int(context.profile.risk_level)  # 1~5
            # 1→0.4, 3→1.0, 5→1.5 线性
            scale = 0.4 + (rl_num - 1) * 0.275
            scores = {k: float(np.clip(v * scale, -1.0, 1.0)) for k, v in scores.items()}

        avg = float(np.mean(list(scores.values()))) if scores else 0.0
        if avg > 0.06:
            dir = Signal.LONG
        elif avg < -0.06:
            dir = Signal.SHORT
        else:
            dir = Signal.NEUTRAL
        conf = float(np.clip(0.6 + abs(avg) * 0.4, 0.5, 0.95))
        reasoning = [
            f"组合经理校准：共吸收 {len(prev)} 位分析师意见，综合分={avg:+.2f}",
            f"结合画像约束乘子后方向={dir.name}",
        ]
        return AgentOutput(
            agent_name=self.name,
            direction=dir,
            confidence=conf,
            per_asset_scores=scores,
            reasoning=reasoning,
        )
