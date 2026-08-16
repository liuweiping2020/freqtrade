"""
端到端智能投研流程示例 (examples/full_pipeline_demo)
=====================================================

演示从「风险问卷 → 画像 → 资产配置 → 因子加工 → 多智能体决策 →
再平衡触发 → 组合回测 → 绩效报告」的全流程，输出一段 Markdown，
可直接复制进 Freqtrade 的日志 / 前端页面。

运行方式::

    python -m freqtrade.robo_advisor.examples.full_pipeline_demo
"""

from __future__ import annotations

from pprint import pprint

import numpy as np
import pandas as pd

from freqtrade.robo_advisor import (
    # 画像
    RiskQuestionnaire,
    RiskAssessor,
    # 资产配置
    MarkowitzAllocator,
    RiskParityAllocator,
    # 因子
    FactorPurifier,
    FactorAnalyzer,
    FactorCombiner,
    BarraAttributor,
    # 智能体
    Orchestrator,
    ResearchContext,
    FundamentalAnalyst,
    TechnicalsAnalyst,
    RiskController,
    PortfolioManager,
    # 组合
    PortfolioRebalancer,
    # 报告
    ReportEngine,
    # NLP
    SentimentAnalyzer,
    ReportParser,
)


# ---------------------------------------------------------------------------
# 1) 合成数据：价格、因子、新闻
# ---------------------------------------------------------------------------


def make_synthetic_data(seed: int = 42, symbols=("BTC", "ETH", "SOL", "BNB", "USDT"), T=365):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=T, freq="D")
    # 每日收益率：BTC 高波动高收益，USDT 0 收益
    mu = np.array([0.0008, 0.0006, 0.0012, 0.0005, 0.0000])
    sigma = np.array([0.032, 0.035, 0.055, 0.030, 0.0005])
    # 相关矩阵
    corr = np.array([
        [1.00, 0.75, 0.65, 0.55, 0.00],
        [0.75, 1.00, 0.70, 0.60, 0.00],
        [0.65, 0.70, 1.00, 0.50, 0.00],
        [0.55, 0.60, 0.50, 1.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((T, len(symbols)))
    rets = mu.reshape(1, -1) + sigma.reshape(1, -1) * (z @ L.T)
    rets[:, -1] = 0.0  # USDT 保持 0 收益
    prices = 100.0 * np.vstack([np.ones(len(symbols)), (1 + rets).cumprod(axis=0)])
    prices = pd.DataFrame(prices[:-1], index=dates, columns=list(symbols))

    # 因子：3 个风格因子 + 宏观 1 个
    # 动量：前 20 日累计收益
    mom = prices.pct_change(20).shift(1)
    # 价值：历史收益均值的反向（简化 proxy）
    val = -prices.pct_change(60).shift(1)
    # 低波：过去 20 日波动率的反向
    vol20 = prices.pct_change().rolling(20).std().shift(1)
    lowvol = -vol20
    # 宏观：全市场平均（BTC 驱动）
    macro = prices["BTC"].pct_change(5).rolling(10).mean().shift(1)
    factor_df = pd.concat({
        "momentum": mom, "value": val, "lowvol": lowvol,
    }, axis=1).reindex(prices.index)
    # 因子值有部分缺失，用于测试 purifier
    factor_df = factor_df.mask(rng.random(factor_df.shape) < 0.03)
    macro_df = pd.DataFrame({"crypto_macro": macro}).ffill().bfill()

    # 新闻：每天几条文本，带情感倾向
    news_per_day = []
    for d, btc_ret in zip(prices.index, rets[:, 0]):
        pieces = []
        # 好消息
        if btc_ret > 0.02:
            pieces.append(f"{d.date()}: BTC 强势上涨，市场情绪乐观，资金面明显回暖。")
            pieces.append(f"机构资金持续入场，ETF 净流入显著。")
        elif btc_ret < -0.02:
            pieces.append(f"{d.date()}: BTC 承压下跌，投资者避险情绪升温。")
            pieces.append(f"宏观不确定性上升，短期需警惕回调。")
        else:
            pieces.append(f"{d.date()}: 市场窄幅震荡，等待方向选择。")
        pieces.append("分析师建议关注低波防御性资产，控制仓位。")
        news_per_day.append(pieces)
    return {
        "prices": prices, "factor_df": factor_df, "macro": macro_df,
        "news_per_day": news_per_day, "rets": pd.DataFrame(rets, index=dates, columns=list(symbols)),
    }


# ---------------------------------------------------------------------------
# 2) 主流程
# ---------------------------------------------------------------------------


def run_demo() -> None:
    print("=" * 70)
    print(" Freqtrade 智能投研系统 · 端到端 Demo")
    print("=" * 70)

    data = make_synthetic_data()
    prices = data["prices"]
    syms = list(prices.columns)

    # ---- 2.1 风险问卷 → 画像 ---------------------------------------
    print("\n【1】风险评估问卷")
    answers = RiskQuestionnaire.sample_answers(seed=7)
    assessor = RiskAssessor()
    profile = assessor.evaluate(answers, investment_horizon="long",
                                financial_goal="steady_growth")
    profile = profile.derive_constraints(override=True)
    print(f"  问卷得分：{profile.risk_score:.0f}/10  →  风险等级：{profile.risk_level}")
    print(f"  最大可接受回撤：{profile.max_drawdown_tolerance:.0%}，最大杠杆：{profile.suggested_max_leverage:.2f}x")
    mix = profile.suggest_asset_mix()
    print(f"  建议资产混合：{ {k: f'{v:.0%}' for k,v in mix.items()} }")

    # ---- 2.2 资产配置 ----------------------------------------------
    print("\n【2】资产配置：等权 / 风险平价 / Markowitz 对比")
    # 最近 180 日协方差
    rets_est = prices.pct_change().dropna().iloc[-180:]
    cov = rets_est.cov().to_numpy() * 252
    mu_est = rets_est.mean().to_numpy() * 252

    alloc_rp = RiskParityAllocator(
        default_max_weight=0.60, default_min_weight=0.0,
    )
    alloc_group_constraints = [
        {"assets": [s for s in syms if s != "USDT"], "min": 0.30, "max": 0.80},
    ]
    w_rp_res = alloc_rp.allocate(
        assets=syms, cov_matrix=cov, expected_returns=None,
        group_constraints=alloc_group_constraints,
    )
    w_rp = dict(zip(w_rp_res.assets, w_rp_res.weights))
    w_rp_pct = {k: f"{v*100:.1f}%" for k, v in w_rp.items()}
    print(f"  风险平价：σ={w_rp_res.volatility:.2%} 夏普={w_rp_res.sharpe:.2f} 权重：{w_rp_pct}")

    alloc_mv = MarkowitzAllocator(objective="max_sharpe", default_max_weight=0.60)
    w_mv_res = alloc_mv.allocate(
        assets=syms, cov_matrix=cov, expected_returns=mu_est, risk_free_rate=0.04,
        group_constraints=alloc_group_constraints,
    )
    w_mv = dict(zip(w_mv_res.assets, w_mv_res.weights))
    w_mv_pct = {k: f"{v*100:.1f}%" for k, v in w_mv.items()}
    print(f"  Markowitz (MaxSharpe)：μ={w_mv_res.expected_return:.2%} σ={w_mv_res.volatility:.2%} 夏普={w_mv_res.sharpe:.2f} 权重：{w_mv_pct}")

    # 目标权重：取风险平价作为「投顾建议」，Markowitz 作为 benchmark
    target_w = w_rp

    # ---- 2.3 因子预处理 & 合成 & 分析 ------------------------------
    print("\n【3】因子处理：清洗 → 合成 → IC / 分层回测")
    factor_df_raw = data["factor_df"].dropna(how="all").iloc[-180:]
    # 转为面板：index=[date, asset]，columns=factor_names
    # 当前 factor_df_raw.columns 是 MultiIndex (factor_name, symbol)
    factor_panel = factor_df_raw.stack(level=-1)  # index=[date, symbol]，列=factor_names
    factor_panel.index.names = ["date", "asset"]
    factor_panel = factor_panel.sort_index()

    purifier = FactorPurifier(pipeline="default")
    purified = purifier.fit_transform(factor_panel)

    # 构造 forward_returns 面板：下一日收益率（index=[date, asset], 列名=ret_1d）
    # purified 的日期
    dates_unique = purified.index.get_level_values("date").unique().sort_values()
    prices_slice = prices.reindex(dates_unique)
    # shift(-1) 让今天对应明天的收益
    fwd_rets_df = prices_slice.pct_change().shift(-1)
    # stack
    fwd_rets_panel = fwd_rets_df.stack().rename("ret_1d").to_frame()
    fwd_rets_panel.index.names = ["date", "asset"]
    # 对齐到 purified 的 index
    forward_df = fwd_rets_panel.reindex(purified.index).dropna()
    purified_aligned = purified.reindex(forward_df.index)
    forward_series = forward_df["ret_1d"]

    # 注意：测试资产仅 5 个，分位数用 2（避免 qcut duplicates 报错）
    analyzer = FactorAnalyzer(quantiles=2, min_assets_per_date=2)
    # forward_returns 必须是 DataFrame，列名 ret_1d / ret_5d ...
    fr_df = forward_df  # 已经对齐
    # 逐个因子简单诊断
    per_factor_stats = []
    for col in purified_aligned.columns:
        fa_res = analyzer.run(purified_aligned[col], fr_df, factor_name=str(col))
        ics = fa_res.ic_summary()
        qp = fa_res.quantile_performance()
        # 多空年化 = 最高分组 - 最低分组 的 annual_return
        qnames = sorted([q for q in qp.index if q.startswith("Q")], key=lambda s: int(s.replace("Q", "")))
        if len(qnames) >= 2:
            ls_mean = float(qp.loc[qnames[-1], "annual_return"] - qp.loc[qnames[0], "annual_return"])
        else:
            ls_mean = np.nan
        mic = float(ics.get("mean_ic", np.nan))
        ir = float(ics.get("icir", np.nan))
        per_factor_stats.append((str(col), mic, ir, ls_mean))
    print("  各因子表现：")
    for name, mic, ir, ls in per_factor_stats:
        print(f"    · {name:<10s}  mean_ic={mic:+.3f}  ICIR={ir:+.3f}  多空(高-低)年化={ls:+.2%}")

    combiner = FactorCombiner(method="icir_weight", rolling_window=60, warmup=20)
    comb_res = combiner.combine(purified_aligned, forward_returns=forward_series)
    composite = comb_res.combined
    print(f"  合成因子 shape = {composite.shape}，最近一日截面 std = {composite.iloc[-1]:.3f}")
    if "weights_mean" in comb_res.summary:
        print(f"  合成因子平均权重：{ {k: round(v,3) for k,v in comb_res.summary['weights_mean'].items()} }")

    # ---- 2.4 NLP：情感 + 研报解析 ----------------------------------
    print("\n【4】NLP：情感分析 + 研报解析")
    sent = SentimentAnalyzer()
    news_flat = [n for day in data["news_per_day"][-14:] for n in day]
    sents = sent.analyze_batch(news_flat)
    avg_score = np.mean([s.score for s in sents])
    print(f"  最近 14 天新闻共 {len(sents)} 条，平均情感得分 = {avg_score:+.3f}")

    parser = ReportParser()
    sample_report = """
# 2024 年加密市场深度研究报告

## 一、宏观环境
2024 年全球市场受美联储降息预期推动，风险资产普遍上涨。BTC 上半年累计上涨超过 50%。
主要驱动来自 ETF 资金持续流入，机构持仓比例从 5% 提升至 12%。

## 二、产业链分析
算力板块保持高景气，BTC 全网算力环比增长 15%。Layer2 生态活跃地址数同比增长 120%。

## 三、投资建议
建议维持超配 BTC/ETH，目标价分别为 80,000 美元和 4,200 美元，
投资评级：买入（分析师：张明 / 首席）。
"""
    parse_res = parser.parse(sample_report, title="2024 加密市场深度")
    print(f"  研报摘要：{parse_res.abstract[:80]}…")
    print(f"  关键词 Top5：{parse_res.keywords[:5]}")
    print(f"  抽取实体：评级={parse_res.entities['ratings']}  资产={parse_res.entities['mentioned_assets'][:8]}  目标价={parse_res.entities['target_prices']}")
    print(f"  研报情感得分：{parse_res.sentiment_score:+.3f}")

    # ---- 2.5 多智能体决策 -----------------------------------------
    print("\n【5】多智能体决策（Orchestrator · sequential）")
    # 构造 ResearchContext：prices 是 {symbol: DataFrame} 格式，方便各 Agent 独立查
    ctx_window = 60
    prices_dict: dict[str, pd.DataFrame] = {}
    for s in syms:
        sdf = pd.DataFrame({
            "open": prices[s].iloc[-ctx_window:],
            "high": prices[s].iloc[-ctx_window:] * (1 + np.abs(np.random.default_rng(hash(s)&0xffffffff).normal(0, 0.005, ctx_window))),
            "low":  prices[s].iloc[-ctx_window:] * (1 - np.abs(np.random.default_rng(hash(s+s)&0xffffffff).normal(0, 0.005, ctx_window))),
            "close": prices[s].iloc[-ctx_window:],
            "volume": np.random.default_rng(hash(s)&0xffffffff).integers(1000, 100000, size=ctx_window).astype(float),
        }, index=prices.index[-ctx_window:])
        prices_dict[s] = sdf
    # factor_scores：把最后一日的 purified_aligned 截面转成 {factor_name:{symbol:score}}
    last_date = purified_aligned.index.get_level_values("date").max()
    last_scores_xs = purified_aligned.xs(last_date, level="date")
    factor_scores_dict = {col: last_scores_xs[col].to_dict() for col in last_scores_xs.columns}
    # 由价格判断大概的 regime（trend / range / high_vol）
    recent_ret = prices.pct_change().iloc[-20:].mean(axis=1)
    if float(recent_ret.std() * np.sqrt(365)) > 0.6:
        regime = "high_vol"
    elif abs(float(recent_ret.sum())) > 0.05:
        regime = "trend"
    else:
        regime = "range"
    ctx = ResearchContext(
        symbols=syms,
        date=prices.index[-1],
        market_regime=regime,
        prices=prices_dict,
        factor_scores=factor_scores_dict,
        profile=profile,
        allocation=target_w,
        extras={
            "macro": data["macro"].iloc[-ctx_window:].copy() if data["macro"] is not None else None,
            "sentiment_by_symbol": {s: float(np.clip(avg_score, -1, 1)) for s in syms},
            "data_backend": "synthetic_demo",
        },
    )
    orchestrator = Orchestrator(
        mode="sequential",
        debate_rounds=2,
    )
    orchestrator.register(FundamentalAnalyst(), weight=1.0)
    orchestrator.register(TechnicalsAnalyst(), weight=1.0)
    orchestrator.register(RiskController(), weight=1.0)
    orchestrator.register(PortfolioManager(), weight=1.2)

    decision = orchestrator.run(ctx)
    print(f"  最终投资决策：总体方向={decision.overall_direction.name}，整体置信={decision.aggregate_confidence:.2f}")
    # 资产排序
    ranked = sorted(decision.per_asset_scores.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  推荐排序（前 3）：{[s for s, _ in ranked[:3]]}")
    w_pct = {k: f"{v*100:.1f}%" for k, v in decision.target_weights.items() if v > 0.01}
    print(f"  建议权重：{w_pct}")
    if decision.summary:
        print(f"  决策摘要：{decision.summary[:160]}…")

    # ---- 2.6 再平衡触发 ------------------------------------------
    print("\n【6】再平衡：当前权重 → 目标权重")
    # 构造「当前权重」：偏离 target_w（模拟漂移）
    drift = np.array([0.12, 0.01, -0.05, -0.03, -0.05])
    current_w = {s: max(0.0, target_w[s] + drift[i]) for i, s in enumerate(syms)}
    # 归一
    total = sum(current_w.values())
    current_w = {s: v / total for s, v in current_w.items()}

    rebalancer = PortfolioRebalancer(
        threshold_individual=0.05, threshold_portfolio=0.10,
        schedule_days=30, trading_cost_bps=15, mode="threshold+schedule",
    )
    pxs_last = {s: float(prices[s].iloc[-1]) for s in syms}
    orders, info = rebalancer.run(
        current_weights=current_w, target_weights=target_w,
        current_prices=pxs_last, total_portfolio_value_usdt=100_000.0,
        last_rebalance_ts=pd.Timestamp.now().timestamp() - 45 * 86400,  # 45 天前
    )
    print(f"  是否触发：{info.triggered}  原因：{info.trigger_reasons}")
    print(f"  单资产最大偏离：{info.individual_max_deviation:.2%}  组合整体偏离：{info.portfolio_deviation:.2%}")
    print(f"  成本估算 ≈ ${info.estimated_trading_cost_usdt:.2f}，收益估算 ≈ ${info.estimated_gain_from_rebalance:.2f}")
    if orders:
        print("  订单（先卖后买）：")
        for o in orders:
            sign = "+" if o.side.name == "BUY" else "-"
            print(f"    · {o.side.name:4s} {o.symbol:5s}  {sign}{abs(o.notional_usdt):8.2f} USDT  (权重 {o.weight_delta*100:+.1f}%)")

    # ---- 2.7 组合回测 & 绩效报告 -----------------------------------
    print("\n【7】组合回测 & 绩效报告（再平衡执行目标权重）")
    # 构造目标权重时序：每月最后一天调仓
    target_series = pd.DataFrame(
        [target_w] * len(prices), index=prices.index, columns=syms,
    )
    # 让权重在每月末重置
    month_ends = prices.resample("ME").last().index
    for i in range(1, len(month_ends)):
        prev_eom = month_ends[i - 1]
        cur_eom = month_ends[i]
        # 给这段时间加一些漂移：目标权重 + 随机噪声，月底再回来
        mask = (prices.index > prev_eom) & (prices.index <= cur_eom)
        rng = np.random.default_rng(i * 17)
        noise = rng.normal(0, 0.015, size=len(syms))
        drift_arr = np.array([target_w[s] for s in syms]) + noise
        drift_arr = np.clip(drift_arr, 0.0, 0.70)
        drift_arr = drift_arr / drift_arr.sum()
        target_series.loc[mask] = drift_arr
        target_series.loc[cur_eom] = [target_w[s] for s in syms]

    equity, infos = rebalancer.backtest_rebalance(target_series, prices, initial_value=100_000.0)
    engine = ReportEngine(periods_per_year=365, risk_free_rate=0.04)
    summary = engine.summarize(equity)
    md = engine.to_markdown(summary, title="Demo 组合绩效报告（再平衡策略）")
    print(md)

    # ---- 2.8 Brinson 归因 & 风险分解 ------------------------------
    print("\n【8】Brinson 归因 & 风险因子分解")
    # 基准权重：等权
    eq_w = pd.DataFrame(1.0 / len(syms), index=target_series.index, columns=syms)
    asset_rets = prices.pct_change().reindex(target_series.index).fillna(0.0)
    br = engine.brinson(target_series, eq_w, asset_rets)
    print(f"  相对等权基准累计主动收益 = {br.total_active:.2%}")
    print(f"   - 配置效应：{br.cumulative.get('allocation',0):.2%}")
    print(f"   - 选股效应：{br.cumulative.get('selection',0):.2%}")
    print(f"   - 交互效应：{br.cumulative.get('interaction',0):.2%}")

    # 因子协方差（用三个风格因子的相关估计）
    F = purified.iloc[-120:].cov().to_numpy(dtype=float) * 252
    factor_names = list(purified.columns.get_level_values(0).unique()) if purified.columns.nlevels > 1 else list(purified.columns)
    # 简化：组合对每个因子按持有资产平均暴露
    w_map = {f: 1.0 / len(factor_names) for f in factor_names}
    rd = engine.risk_decomposition(w_map, F, factor_names=factor_names)
    print(f"  组合年化因子波动率 ≈ {rd.total_volatility:.2%}")
    print(f"  因子风险贡献比例：{ {k:f'{v:.0%}' for k,v in rd.factor_contrib_pct.items()} }")

    print("\n" + "=" * 70)
    print(" 端到端 Demo 完成。可接入真实价格/因子/新闻数据用于生产。")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
