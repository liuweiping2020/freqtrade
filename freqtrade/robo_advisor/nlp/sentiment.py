"""
金融 NLP 情感分析 (nlp/sentiment)
===================================

统一接口，三种后端按优先级自动选择（无需手动选）：

    1) huggingface transformers: "ProsusAI/finbert" / "shiyu-coder/Kronos-7B"
    2) 本地 LLM client（由上游传入）
    3) 词典法 fallback（中英文金融情感词典 + 规则 + 否定词处理）—— 零依赖

统一输出：
    SentimentResult = {
        label:   "positive" / "negative" / "neutral"
        score:   [-1, +1]
        probs:   {positive: 0.1, neutral: 0.2, negative: 0.7}
        model:   使用的后端名
    }

示例::

    sa = SentimentAnalyzer(model_name="ProsusAI/finbert", device="cpu")
    res = sa.analyze("BTC ETF净买入突破10亿，矿工继续增持")
    print(res.score, res.label)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


@dataclass
class SentimentResult:
    text: str
    label: Literal["positive", "negative", "neutral"]
    score: float                                  # -1..+1
    probs: dict[str, float] = field(default_factory=dict)
    model: str = "lexicon"
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 词典法 fallback（中英文双语金融情感词库）
# ---------------------------------------------------------------------------


_POS_ZH = [
    "增长", "上涨", "利好", "突破", "增持", "买入", "超预期", "创新高", "净流入",
    "盈利", "回购", "分红", "降级", "升级", "乐观", "复苏", "扩张", "改善",
    "上升", "稳定", "强劲", "修复", "繁荣", "抄底", "流入", "新高", "爆发",
    "突破", "扩张", "增持", "红利", "龙头", "低估", "加仓", "反转",
]
_NEG_ZH = [
    "下跌", "暴跌", "亏损", "利空", "违约", "破产", "减持", "抛售", "降级",
    "恐慌", "风险", "衰退", "暴跌", "回撤", "亏损", "紧缩", "暴跌", "踩踏",
    "流出", "破位", "新低", "爆仓", "清算", "暴跌", "回调", "下跌", "暴跌",
    "诉讼", "罚款", "违约", "退市", "裁员", "降薪", "坏账", "资不抵债", "欺诈",
    "下跌", "下挫", "走弱", "恶化", "抛售", "崩盘", "滞胀", "衰退", "暴跌",
]
_NEGATE_ZH = ["不", "没", "无", "未", "非", "别", "莫", "否", "难以", "无法", "不会"]

_POS_EN = [
    "surge","rise","gain","bullish","outperform","upgrade","beat","upside","rally",
    "buy","long","growth","profit","dividend","recovery","optimistic","expansion",
    "breakthrough","inflow","support","improve","record high","beat estimates",
    "low","undervalued","opportunity","positive","strong","exceeds","all time high",
]
_NEG_EN = [
    "drop","fall","crash","bearish","downgrade","miss","downside","sell","short",
    "loss","decline","recession","panic","risk","default","liquidation","sell-off",
    "outflow","resistance","collapse","correction","low","overvalued","fine","fined",
    "lawsuit","fraud","delist","bankruptcy","negative","weak","crisis","bubble",
    "miss estimates","bad debt","cut","layoff","pullback","dump","volatility",
]
_NEGATE_EN = ["not","no","none","never","neither","nor","cannot","unlikely","without"]


def _sentiment_by_lexicon(text: str) -> tuple[float, dict[str, int]]:
    """
    返回 [-1, +1] 加权分数 + 匹配计数明细。
    处理规则：
      - 词频匹配（支持中文子串匹配，英文空白分隔加正则）
      - 否定词前 3 个词范围内的情感词翻转 50% 符号
    """
    t = text
    low = t.lower()
    # --- 中文 ---
    pos_count = 0
    neg_count = 0
    for w in _POS_ZH:
        pos_count += t.count(w)
    for w in _NEG_ZH:
        neg_count += t.count(w)
    # --- 英文 ---
    for w in _POS_EN:
        pos_count += len(re.findall(r"\b" + re.escape(w) + r"\b", low))
    for w in _NEG_EN:
        neg_count += len(re.findall(r"\b" + re.escape(w) + r"\b", low))
    # 否定词简单处理：中文「不*利好」→ 扣减
    for neg_w in _NEGATE_ZH:
        # 扫一遍否定词，看窗口内有无正向/负向词：出现就抵消 1 个
        idx = 0
        while True:
            i = t.find(neg_w, idx)
            if i < 0:
                break
            idx = i + 1
            window = t[i: i + 12]
            hit_pos = sum(1 for w in _POS_ZH if w in window)
            hit_neg = sum(1 for w in _NEG_ZH if w in window)
            if hit_pos:
                pos_count = max(pos_count - 1, 0)
                neg_count += 1  # 正向→负向
            if hit_neg:
                neg_count = max(neg_count - 1, 0)
                pos_count += 1
    for neg_w in _NEGATE_EN:
        pattern = re.compile(r"\b" + re.escape(neg_w) + r"\b\s+(\w+\s+){0,2}(\w+)", re.IGNORECASE)
        for m in pattern.finditer(low):
            tail = m.group(0).lower()
            if any(w in tail for w in _POS_EN):
                pos_count = max(pos_count - 1, 0); neg_count += 1
            if any(w in tail for w in _NEG_EN):
                neg_count = max(neg_count - 1, 0); pos_count += 1
    total = pos_count + neg_count
    if total == 0:
        score = 0.0
    else:
        score = (pos_count - neg_count) / total  # [-1, 1]
    return score, {"positive_matches": pos_count, "negative_matches": neg_count}


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


Backend = Literal["auto", "finbert", "kronos", "llm_client", "lexicon"]


class SentimentAnalyzer:
    """
    金融情感分析统一接口。

    默认 `backend='auto'`：会按以下优先级自动挑选可用后端：
        1. llm_client（构造时传入任何满足 generate(...) 的对象）
        2. transformers + torch → 加载 ProsusAI/finbert
        3. 回退到词典法（零依赖）
    """

    def __init__(
        self,
        backend: Backend = "auto",
        model_name: str = "ProsusAI/finbert",
        llm_client: Any = None,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.llm_client = llm_client
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._pipeline = None
        self._actual_backend: str | None = None
        self._resolve_backend()

    # ---- 后端探测 ------------------------------------------------------

    def _resolve_backend(self) -> None:
        if self.backend == "lexicon":
            self._actual_backend = "lexicon"
            return
        # llm_client 最高优先级（如果指定了 llm_client 或者 backend == llm_client）
        if self.backend == "llm_client" or self.llm_client is not None:
            if self.llm_client is None:
                self._actual_backend = "lexicon"
            else:
                self._actual_backend = "llm_client"
            return
        # transformer 模型
        if self.backend in ("auto", "finbert", "kronos"):
            try:
                from transformers import pipeline  # type: ignore
                import torch  # noqa: F401
                task = "text-classification"
                self._pipeline = pipeline(
                    task, model=self.model_name, device=self.device,
                    truncation=True, max_length=self.max_length,
                    padding=True, top_k=None,
                )
                self._actual_backend = f"transformers:{self.model_name}"
                return
            except Exception:
                self._pipeline = None
        self._actual_backend = "lexicon"

    @property
    def actual_backend(self) -> str:
        return self._actual_backend or "unknown"

    # ---- 主入口 --------------------------------------------------------

    def analyze(self, text: str) -> SentimentResult:
        return self.analyze_batch([text])[0]

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        if self._actual_backend and self._actual_backend.startswith("transformers:") and self._pipeline:
            results = self._via_transformers(texts)
        elif self._actual_backend == "llm_client" and self.llm_client is not None:
            results = self._via_llm(texts)
        else:
            results = self._via_lexicon(texts)
        # 规范：score ∈ [-1, +1]
        for r in results:
            r.score = float(max(-1.0, min(1.0, r.score)))
        return results

    # ---- 各后端 --------------------------------------------------------

    def _via_transformers(self, texts: list[str]) -> list[SentimentResult]:
        assert self._pipeline is not None
        out: list[SentimentResult] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            preds = self._pipeline(batch)
            for text, pred in zip(batch, preds):
                # pred: list[dict(label, score)] 因为 top_k=None
                probs: dict[str, float] = {}
                if isinstance(pred, list):
                    for item in pred:
                        lab = item["label"].lower()
                        if lab in ("positive", "pos"):
                            probs["positive"] = float(item["score"])
                        elif lab in ("negative", "neg"):
                            probs["negative"] = float(item["score"])
                        else:
                            probs["neutral"] = float(item["score"])
                else:
                    probs = {str(pred.get("label")).lower(): float(pred.get("score", 0.0))}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                neu = probs.get("neutral", 0.0)
                probs.setdefault("positive", 0.0)
                probs.setdefault("negative", 0.0)
                probs.setdefault("neutral", 0.0)
                # 规范化三个加起来 = 1
                total = probs["positive"] + probs["negative"] + probs["neutral"]
                if total == 0:
                    probs = {"positive": 1/3, "negative":1/3, "neutral":1/3}
                else:
                    probs = {k: v / total for k, v in probs.items()}
                score = probs["positive"] - probs["negative"]
                if score > 0.1:
                    label = "positive"
                elif score < -0.1:
                    label = "negative"
                else:
                    label = "neutral"
                out.append(SentimentResult(text=text, label=label, score=score, probs=probs,
                                            model=self._actual_backend or ""))
        return out

    def _via_llm(self, texts: list[str]) -> list[SentimentResult]:
        out: list[SentimentResult] = []
        # Prompt 结构化：让 LLM 返回 JSON
        system_prompt = (
            "You are a financial sentiment classifier. "
            "For each given news headline or text, output ONLY a compact JSON object: "
            '{"label":"positive|negative|neutral", "score":-1..1, "probs":{"positive":0.x,"neutral":0.x,"negative":0.x}} '
            "score = prob(positive)-prob(negative). Do not include any other words. Respond in one line per input."
        )
        for t in texts:
            prompt = f"{system_prompt}\n\nTEXT: {t}\nJSON:"
            raw = None
            if hasattr(self.llm_client, "generate"):
                raw = str(self.llm_client.generate(prompt))
            elif callable(self.llm_client):
                raw = str(self.llm_client(prompt))
            import json as _json
            label, score, probs = "neutral", 0.0, {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
            if raw:
                try:
                    # 从原始文本中提取 JSON
                    m = re.search(r"\{[^{}]+\}", raw)
                    if m:
                        data = _json.loads(m.group(0))
                        label = str(data.get("label", label)).lower()
                        score = float(data.get("score", score))
                        if isinstance(data.get("probs"), dict):
                            probs = {k.lower(): float(v) for k, v in data["probs"].items()}
                except Exception:
                    # fallback 简单关键词
                    sc, _ = _sentiment_by_lexicon(t)
                    score = float(sc)
            else:
                sc, _ = _sentiment_by_lexicon(t)
                score = float(sc)
            out.append(SentimentResult(text=t, label=label, score=score, probs=probs, model="llm_client"))
        return out

    def _via_lexicon(self, texts: list[str]) -> list[SentimentResult]:
        out: list[SentimentResult] = []
        for t in texts:
            score, meta = _sentiment_by_lexicon(t)
            p = (score + 1.0) / 2.0  # -1..1 → 0..1
            if score > 0.1:
                label, pos, neu, neg = "positive", p, 0.3, 1 - p
            elif score < -0.1:
                label, pos, neu, neg = "negative", 1 - p, 0.3, p
            else:
                label, pos, neu, neg = "neutral", 0.25, 0.5, 0.25
            total = pos + neu + neg
            probs = {"positive": pos/total, "neutral": neu/total, "negative": neg/total}
            out.append(SentimentResult(text=t, label=label, score=score, probs=probs,
                                        model="lexicon", meta=meta))
        return out
