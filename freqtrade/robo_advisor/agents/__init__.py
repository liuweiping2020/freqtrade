"""多智能体投研框架子模块 (agents).

包含：
    - base:       智能体基类 (BaseAgent) 与编排器 (Orchestrator)
    - analysts:   职能型投研智能体（基本面/技术面/舆情/风控/组合）
"""

from __future__ import annotations

__all__ = ["BaseAgent", "Orchestrator"]


def __getattr__(name):
    import importlib

    if name in ("BaseAgent", "Orchestrator"):
        module = importlib.import_module(".base", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
