"""金融 NLP 子模块 (nlp).

包含：
    - sentiment:     情感分析接口层 (FinBERT / Kronos / 通用 LLM 适配)
    - report_parser: 研报理解与结构化信息提取
"""

from __future__ import annotations

__all__ = ["SentimentAnalyzer", "ReportParser"]


def __getattr__(name):
    import importlib

    if name == "SentimentAnalyzer":
        module = importlib.import_module(".sentiment", __name__)
        return getattr(module, name)
    if name == "ReportParser":
        module = importlib.import_module(".report_parser", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
