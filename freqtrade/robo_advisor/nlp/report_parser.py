"""
研报理解与结构化信息提取 (nlp/report_parser)
================================================

零 NLP 模型依赖版本：用「词典匹配 + 正则实体 + 抽取式 TextRank 摘要 + 结构切分」
快速提取研报核心信息。如提供 llm_client，可升级为 LLM 版摘要/结构化问答。

功能：
    1) 段落切分：标题、摘要、正文、章节、结尾（按 markdown/章节编号 + 关键词启发）
    2) 关键信息实体：
        - 公司名/代码
        - 人名（分析师/高管）
        - 日期
        - 金额/估值
        - 投资评级 / 目标价（Buy/Sell/Hold、目标价、上调/下调）
    3) 关键词抽取：TF-IDF（相对通用停用词语料）+ 金融领域关键词词典
    4) 抽取式摘要：TextRank 风格 BM25 相似度图 + PageRank 选前 N 句
    5) 情感总分：调用 SentimentAnalyzer 或 词典平均

典型用法::

    parser = ReportParser()
    res = parser.parse(report_text, title="2024 比特币行业深度报告")
    print(res.entities)          # 公司/评级/金额…
    print(res.keywords[:20])     # 关键词
    print(res.abstract)          # 3 句摘要
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 停用词 & 关键词词典（金融）
# ---------------------------------------------------------------------------


_STOPWORDS_ZH = set(list("的一是在了不有和人就都一上个国到说以而及与但也并都或又你我他它这那这样那样")
    + "".split() + ["我们", "你们", "他们", "它们", "这是", "那是", "以及", "对于", "关于", "由于",
                    "因此", "所以", "然后", "之后", "之前", "但是", "不过", "只是", "可以", "可能",
                    "应该", "需要", "进行", "通过", "一种", "一个", "等", "等等", "根据", "目前",
                    "同时", "主要", "方面", "其中", "相关", "如果", "虽然", "因为", "所以", "已经",
                    "其中", "以上", "以下", "以及", "仍将", "包括", "表示", "认为"])

_STOPWORDS_EN = {w.lower() for w in """
a an the and or but if while with for of to in on at from by is are was were be been being
this that these those i you he she it we they them my our your his her its their me him us
not no do does did has have had will would should can could may might so as than then there
here just about also into more out up down over under again further any all some such too only
own same so than too very can just don now
""".split()}

_FIN_KEYWORDS_ZH = [
    "市盈率", "市净率", "市销率", "市现率", "ROE", "ROA", "ROIC", "EPS", "PE", "PB", "PEG",
    "营收", "净利润", "毛利", "毛利率", "净利率", "现金流", "负债率", "资产负债表", "利润表",
    "现金流量表", "自由现金流", "分红", "回购", "目标价", "投资评级", "买入", "增持", "持有",
    "减持", "卖出", "上调", "下调", "超预期", "低预期", "贝塔", "夏普", "阿尔法", "回撤",
    "流动性", "融资", "融券", "大宗交易", "ETF", "公募", "私募", "量化", "对冲", "套利",
    "趋势", "震荡", "突破", "支撑", "压力", "估值", "低估", "高估", "市值", "流通市值",
    "ETF", "BTC", "ETH", "SOL", "BNB", "XRP", "ETF", "比特币", "以太坊", "稳定币",
    "美联储", "加息", "降息", "CPI", "PPI", "GDP", "PMI", "非农", "利率", "国债",
]
_FIN_KEYWORDS_EN = [w.lower() for w in [
    "PE", "PB", "ROE", "ROA", "ROIC", "EPS", "PEG", "EBITDA", "EV", "DCF",
    "revenue", "net income", "gross margin", "operating margin", "cash flow", "FCF",
    "balance sheet", "income statement", "P&L", "dividend", "buyback",
    "target price", "investment rating", "BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL",
    "upgrade", "downgrade", "beat", "miss", "estimate",
    "alpha", "beta", "sharpe", "drawdown", "volatility", "VaR", "Sharpe",
    "ETF", "mutual fund", "hedge fund", "quantitative", "arbitrage", "trend", "range",
    "support", "resistance", "breakout", "valuation", "overvalued", "undervalued",
    "BTC", "ETH", "SOL", "BNB", "XRP", "bitcoin", "ethereum", "stablecoin",
    "Fed", "FOMC", "rate hike", "rate cut", "CPI", "PPI", "GDP", "PMI", "treasury",
]]


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------


@dataclass
class ReportParseResult:
    title: str
    sections: list[dict[str, Any]]              # [{"title":"...", "paragraphs": [...]}]
    abstract: str                               # 抽取式摘要（N 句）
    keywords: list[tuple[str, float]]           # [(term, score), ...] 降序
    entities: dict[str, list[str]]              # {"companies":..., "ratings":..., "target_prices":..., "dates":..., "amounts":..., "analysts":...}
    sentiment_score: float                      # 整篇 [-1, +1]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [{"title": s["title"], "n_paragraphs": len(s["paragraphs"])} for s in self.sections],
            "abstract": self.abstract,
            "keywords": self.keywords[:50],
            "entities": self.entities,
            "sentiment_score": self.sentiment_score,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class ReportParser:
    """研报/长文本结构化抽取器（零模型依赖 + 可选 LLM）。"""

    def __init__(self, llm_client: Any = None, sentence_delimiters: str = "。！？!?.\n") -> None:
        self.llm_client = llm_client
        self.sentence_delimiters = sentence_delimiters

    # --- 主入口 --------------------------------------------------------

    def parse(
        self,
        text: str,
        title: str = "未命名研报",
        max_keywords: int = 50,
        abstract_sentences: int = 3,
    ) -> ReportParseResult:
        if not text:
            return ReportParseResult(title=title, sections=[], abstract="", keywords=[], entities={}, sentiment_score=0.0)
        clean_text = self._normalize(text)
        sections = self._split_sections(clean_text, title)
        # 句子切分
        sentences = self._split_sentences(clean_text)
        # 关键词
        kw = self._extract_keywords(clean_text, top=max_keywords)
        # 实体
        entities = self._extract_entities(clean_text)
        # 摘要（TextRank）
        abst = self._textrank_summary(sentences, top=abstract_sentences)
        # 情感
        sent = self._overall_sentiment(clean_text, sentences)

        return ReportParseResult(
            title=title,
            sections=sections,
            abstract=abst,
            keywords=kw,
            entities=entities,
            sentiment_score=sent,
            meta={"sentences": len(sentences), "chars": len(clean_text)},
        )

    # --- 1) 规范化 -----------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        # 合并多余空白（保留换行，用于章节切分）
        lines = [ln.rstrip() for ln in text.splitlines()]
        # 去除连续空行
        cleaned_lines: list[str] = []
        last_blank = False
        for ln in lines:
            is_blank = (ln.strip() == "")
            if is_blank and last_blank:
                continue
            cleaned_lines.append(ln)
            last_blank = is_blank
        return "\n".join(cleaned_lines)

    # --- 2) 章节切分 ---------------------------------------------------

    _TITLE_PATTERNS = [
        re.compile(r"^\s*#{1,6}\s+(.+)$"),                                         # Markdown
        re.compile(r"^\s*([一二三四五六七八九十百千]+、\s*.+)$"),                     # 中文一、二、
        re.compile(r"^\s*(\d+(\.\d+)*\s+[A-Za-z\u4e00-\u9fa5].*)$"),                # 1. 2.1
        re.compile(r"^\s*[（(]([一二三四五六七八九十百千\d]+)[)）]\s*.+$"),            # （一）(1)
        re.compile(r"^\s*第[一二三四五六七八九十百千]+[章节部分篇条].*$"),             # 第一章 / 第二节
    ]

    @classmethod
    def _is_section_title(cls, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        for p in cls._TITLE_PATTERNS:
            if p.match(s):
                return True
        # 启发：如果这一行不长且全是中文/英文且无标点，且前后有空行 → 视为标题
        if len(s) <= 40 and not re.search(r"[。，,；;！？!?]", s):
            return True
        return False

    @classmethod
    def _split_sections(cls, text: str, root_title: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current_title = root_title
        current_paras: list[str] = []
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln:
                if current_paras and current_paras[-1] != "":
                    current_paras.append("")
                continue
            if cls._is_section_title(ln) and (not current_paras or current_paras[-1] == ""):
                # 新章节
                if current_title and any(p.strip() for p in current_paras):
                    sections.append({
                        "title": current_title,
                        "paragraphs": [p for p in current_paras if p.strip()],
                    })
                current_title = ln.strip()
                current_paras = []
            else:
                current_paras.append(ln)
        # 尾部
        if current_title and any(p.strip() for p in current_paras):
            sections.append({
                "title": current_title,
                "paragraphs": [p for p in current_paras if p.strip()],
            })
        if not sections:
            sections = [{"title": root_title, "paragraphs": [text]}]
        return sections

    # --- 3) 句子切分 ---------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        # 先去换行 → 切
        flat = re.sub(r"[\r\n]+", " ", text)
        # 按分隔符切，但保留符号
        pattern = re.compile(rf"(?<=[{re.escape(self.sentence_delimiters)}])\s+")
        sents = [s.strip() for s in pattern.split(flat) if s.strip()]
        return sents

    # --- 4) 关键词抽取 -------------------------------------------------

    def _extract_keywords(self, text: str, top: int) -> list[tuple[str, float]]:
        # tokens：中文单字 + 英文词 + 数字
        tokens: list[str] = []
        # 英文/数字用正则直接提
        for m in re.finditer(r"[A-Za-z]{2,}[\w\-+/.]*|[\d.]+%|[\d.]+", text):
            t = m.group(0)
            if t.lower() in _STOPWORDS_EN:
                continue
            tokens.append(t.lower() if not t.isupper() else t)  # 保留大写的 BTC/ROE 等
        # 中文：1-gram + 2-gram，过滤停用词
        zh_lines = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
        for ch in zh_lines:
            if ch == " " or ch in _STOPWORDS_ZH:
                continue
            tokens.append(ch)
        for i in range(len(zh_lines) - 1):
            bigram = zh_lines[i:i+2]
            if " " in bigram or (bigram[0] in _STOPWORDS_ZH and bigram[1] in _STOPWORDS_ZH):
                continue
            tokens.append(bigram)
        # 金融关键词词典加权 × 2.0
        counter: Counter[str] = Counter(tokens)
        weights: dict[str, float] = {}
        for tok, freq in counter.items():
            boost = 1.0
            if tok in _FIN_KEYWORDS_ZH or tok in _FIN_KEYWORDS_EN:
                boost = 3.0
            elif re.fullmatch(r"[\d.]+", tok):
                boost = 0.2
            weights[tok] = freq * boost
        # IDF 启发式：长度越长 → 越稀缺 → 再乘 log(len+1)
        for tok in list(weights.keys()):
            weights[tok] = weights[tok] * (1 + np.log(len(tok) + 1))
        ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        # 归一化分数
        if not ranked:
            return []
        max_w = ranked[0][1]
        return [(t, round(w / max_w, 4)) for t, w in ranked[:top]]

    # --- 5) 实体识别 ---------------------------------------------------

    _RE_DATE = re.compile(
        r"(20\d{2}[-/年.]\s*\d{1,2}[-/月.]\s*\d{1,2}[日号]?)"
        r"|(\d{1,2}\s*月\s*\d{1,2}[日号]?)"
        r"|(20\d{2}\s*Q[1-4])"
        r"|(20\d{2}[-/年]?\s*上半年|下半年|全年|财年)"
    )
    _RE_AMOUNT = re.compile(
        r"([±+\-]?\s*[\d,.]+\s*(?:%|百分[之点]|基点|[万亿]元|美元|USDT|亿元|元|万|亿|x|倍|bps|BP))"
        r"|(\$\s*[\d,.]+)"
        r"|(¥\s*[\d,.]+)"
    )
    _RE_TARGET_PRICE = re.compile(
        r"(目标价|TP)\s*(?:=|为|：|:)?\s*(?:[¥$\s]*)?([\d.]+)\s*(?:美元|USDT|元)?",
        re.IGNORECASE,
    )
    _RE_RATING_ZH = re.compile(
        r"(买入|增持|强烈推荐|推荐|持有|中性|减持|卖出|观望|谨慎推荐|审慎推荐|超配|标配|低配)"
        r"\s*(?:评级|投资评级|目标)?"
    )
    _RE_RATING_EN = re.compile(
        r"\b(BUY|SELL|HOLD|STRONG[_ ]?BUY|OVERWEIGHT|UNDERWEIGHT|OUTPERFORM|UNDERPERFORM|MARKET[_ ]?PERFORM|EQUAL[_ ]?WEIGHT)\b",
        re.IGNORECASE,
    )
    _RE_CODE = re.compile(r"\b([A-Z]{1,5}[\-]?\d{3,6}|\d{6}\.S[HZ]|\d{5}\.HK|[A-Z]{2,10}-\w{2,10})\b")
    _RE_ANALYST = re.compile(r"([\u4e00-\u9fa5A-Za-z]{2,8})\s*(?:分析师|Analyst|研究员|策略师|首席|CFA|CPA)", re.IGNORECASE)

    @staticmethod
    def _normalize_entity(lst: list[str]) -> list[str]:
        # 去空白、去完全重复、按首次出现顺序（保留最高频）
        seen: set[str] = set()
        out: list[str] = []
        for s in lst:
            v = s.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        dates = [m.group(0) for m in self._RE_DATE.finditer(text)]
        amounts = [m.group(0) for m in self._RE_AMOUNT.finditer(text)]
        target_prices = [f"{m.group(1)}={m.group(2)}" for m in self._RE_TARGET_PRICE.finditer(text)]
        ratings_zh = [m.group(1) for m in self._RE_RATING_ZH.finditer(text)]
        ratings_en = [m.group(1).upper() for m in self._RE_RATING_EN.finditer(text)]
        codes = [m.group(0) for m in self._RE_CODE.finditer(text)]
        analysts = [m.group(1) for m in self._RE_ANALYST.finditer(text)]

        # 公司名/资产名：匹配 BTC/ETH 等资产名 + 常见金融领域公司
        assets = []
        for kw in _FIN_KEYWORDS_ZH + _FIN_KEYWORDS_EN:
            if len(kw) < 2:
                continue
            if isinstance(kw, str) and (kw.upper() in text or kw in text):
                assets.append(kw)

        return {
            "dates": self._normalize_entity(dates)[:20],
            "amounts": self._normalize_entity(amounts)[:30],
            "target_prices": self._normalize_entity(target_prices)[:10],
            "ratings": self._normalize_entity(ratings_zh + ratings_en)[:10],
            "codes": self._normalize_entity(codes)[:15],
            "analysts": self._normalize_entity(analysts)[:10],
            "mentioned_assets": self._normalize_entity(assets)[:30],
        }

    # --- 6) TextRank 摘要 ----------------------------------------------

    def _textrank_summary(self, sentences: list[str], top: int) -> str:
        if not sentences:
            return ""
        if len(sentences) <= top:
            return " ".join(sentences)
        # 词袋 → 句子相似度（BM25 简化版：Jaccard of tokens）
        n = len(sentences)
        # 切 token：中英文混合
        def tokenize(s: str) -> set[str]:
            tokens: set[str] = set()
            for m in re.finditer(r"[A-Za-z]{2,}|[\d.]+%?", s):
                tokens.add(m.group(0).lower())
            for ch in s:
                if "\u4e00" <= ch <= "\u9fa5":
                    tokens.add(ch)
            for i in range(len(s) - 1):
                bigram = s[i:i+2]
                if all("\u4e00" <= c <= "\u9fa5" for c in bigram):
                    tokens.add(bigram)
            return tokens
        sets = [tokenize(s) for s in sentences]
        # 相似度矩阵 W
        W = np.zeros((n, n), dtype=float)
        for i in range(n):
            si = sets[i]
            if len(si) == 0:
                continue
            for j in range(i + 1, n):
                sj = sets[j]
                if len(sj) == 0:
                    continue
                inter = len(si & sj)
                uni = len(si | sj)
                sim = 0.0 if uni == 0 else inter / uni
                if sim > 0.0:
                    W[i, j] = sim
                    W[j, i] = sim
        # PageRank 迭代
        W_row = W.sum(axis=1, keepdims=True)
        W_norm = np.divide(W, W_row, out=np.zeros_like(W), where=W_row != 0)
        d = 0.85
        score = np.full(n, 1.0 / n)
        for _ in range(30):
            new_score = (1 - d) / n + d * (W_norm.T @ score)
            if np.allclose(new_score, score, atol=1e-6):
                score = new_score
                break
            score = new_score
        # 取前 top（且保留原文顺序）
        order = np.argsort(-score)[:top]
        order_sorted = sorted(order.tolist())
        return " ".join(sentences[k] for k in order_sorted)

    # --- 7) 整篇情感 ---------------------------------------------------

    @staticmethod
    def _overall_sentiment(text: str, sentences: list[str]) -> float:
        # 调 lexicon 情感分析的内部函数（避免循环 import）
        from .sentiment import _sentiment_by_lexicon
        if not sentences:
            s, _ = _sentiment_by_lexicon(text)
            return s
        scores = []
        for s in sentences[:100]:  # 抽样前 100 句即可
            sc, _ = _sentiment_by_lexicon(s)
            scores.append(sc)
        return float(np.mean(scores)) if scores else 0.0
