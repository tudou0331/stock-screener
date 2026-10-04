"""
analyst_ai.py  —  分析师预期解读 + AI 反向审计 + 判断记账

接在 oversold_quality_screener.py 之后运行:
  python analyst_ai.py                 # 读取 reports/ 下最新的 screen_*.csv，分析前 N 只
  python analyst_ai.py META UBER       # 指定股票
  python analyst_ai.py --evaluate      # 回看过去 AI 判断的实际表现

免费模式（不需要任何 API Key）:
  python analyst_ai.py --prompt [代码…]  # 生成可直接复制粘贴的提示词 reports/prompts_日期.md
  → 把每段提示词贴进 Claude.ai / ChatGPT（打开联网搜索），把它回复的 JSON 存成 reports/replies/代码.json
  python analyst_ai.py --import         # 读入所有回复，写入 ledger 并生成 HTML 报告

三层:
  A. 分析师在想什么（量化）: 一致预期、分歧度、修正动量、历史超预期率、目标价是否在"追价格"
  B. 三方增长对比: 历史增长 vs 分析师预期增长 vs 当前股价隐含增长(反向DCF)
  C. AI 审计员 (Claude + 网络搜索): 逐条检查一致预期背后的关键假设，判断分析师偏乐观/偏悲观/合理
  D. 记账: 每次 AI 判断连同当日价格写入 ledger.csv，事后对比 3/6 个月收益，检验 AI 的分歧是否真的有价值

安装:  pip install yfinance pandas numpy anthropic
环境变量:  export ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import sys
import json
import glob
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CFG = {
    "top_n": 8,                       # 每次送 AI 分析的股票数（控制成本）
    "model": "claude-sonnet-5-5",     # 想要更深入可换 "claude-opus-5-5"
    "max_searches": 8,                # 每只股票 AI 最多搜索次数
    "discount_rate": 0.09,            # 反向 DCF 折现率
    "terminal_growth": 0.025,
    "dcf_years": 10,
    "out_dir": "reports",
    "ledger": "reports/ledger.csv",
}


# ============================== A. 分析师数据 ==============================
def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _loc(df, idx, col):
    try:
        v = df.loc[idx, col]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def analyst_snapshot(ticker):
    t = yf.Ticker(ticker)
    info = _safe(lambda: t.info, {})
    ee = _safe(lambda: t.earnings_estimate, pd.DataFrame())
    re_ = _safe(lambda: t.revenue_estimate, pd.DataFrame())
    tr = _safe(lambda: t.eps_trend, pd.DataFrame())
    rv = _safe(lambda: t.eps_revisions, pd.DataFrame())
    eh = _safe(lambda: t.earnings_history, pd.DataFrame())
    pt = _safe(lambda: t.analyst_price_targets, {})
    rs = _safe(lambda: t.recommendations_summary, pd.DataFrame())
    ud = _safe(lambda: t.upgrades_downgrades, pd.DataFrame())
    inc = _safe(lambda: t.financials, pd.DataFrame())
    qcf = _safe(lambda: t.quarterly_cashflow, pd.DataFrame())

    s = {"ticker": ticker, "name": info.get("shortName", ticker), "sector": info.get("sector"),
         "price": info.get("currentPrice") or info.get("regularMarketPrice"),
         "market_cap": info.get("marketCap"),
         "net_debt": (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)}

    # 一致预期与分歧度
    for per in ("0y", "+1y"):
        avg, lo, hi = (_loc(ee, per, c) for c in ("avg", "low", "high"))
        s[f"eps_{per}"] = avg
        s[f"eps_growth_{per}"] = _loc(ee, per, "growth")
        s[f"n_analysts_{per}"] = _loc(ee, per, "numberOfAnalysts")
        s[f"eps_dispersion_{per}"] = (hi - lo) / abs(avg) if None not in (avg, lo, hi) and avg else None
        s[f"rev_growth_est_{per}"] = _loc(re_, per, "growth")
        cur, d30, d90 = (_loc(tr, per, c) for c in ("current", "30daysAgo", "90daysAgo"))
        s[f"eps_rev30_{per}"] = cur / d30 - 1 if cur and d30 and d30 > 0 else None
        s[f"eps_rev90_{per}"] = cur / d90 - 1 if cur and d90 and d90 > 0 else None
    s["revisions_up30"] = _loc(rv, "+1y", "upLast30days")
    s["revisions_down30"] = _loc(rv, "+1y", "downLast30days")

    # 历史超预期：分析师过去是系统性偏低还是偏高
    if isinstance(eh, pd.DataFrame) and "surprisePercent" in eh.columns and len(eh):
        sp = eh["surprisePercent"].dropna()
        s["beat_rate_4q"] = float((sp > 0).mean()) if len(sp) else None
        s["avg_surprise_4q"] = float(sp.mean()) if len(sp) else None

    # 目标价与评级
    if isinstance(pt, dict) and pt.get("mean") and s["price"]:
        s["pt_mean"], s["pt_low"], s["pt_high"] = pt.get("mean"), pt.get("low"), pt.get("high")
        s["pt_upside"] = pt["mean"] / s["price"] - 1
    if isinstance(rs, pd.DataFrame) and len(rs):
        r0 = rs.iloc[0]
        tot = sum(float(r0.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
        if tot:
            s["pct_buy"] = (float(r0.get("strongBuy", 0)) + float(r0.get("buy", 0))) / tot
            s["pct_sell"] = (float(r0.get("sell", 0)) + float(r0.get("strongSell", 0))) / tot

    # 近 90 天评级/目标价动作：目标价下调是在"追价格"还是有新信息？
    if isinstance(ud, pd.DataFrame) and len(ud):
        u = ud.copy()
        u.index = pd.to_datetime(u.index).tz_localize(None)
        u = u[u.index >= pd.Timestamp.today() - pd.Timedelta(days=90)]
        s["downgrades_90d"] = int((u.get("Action", pd.Series(dtype=str)) == "down").sum())
        s["upgrades_90d"] = int((u.get("Action", pd.Series(dtype=str)) == "up").sum())
        s["pt_cuts_90d"] = int((u.get("priceTargetAction", pd.Series(dtype=str)) == "Lowers").sum())
        s["pt_raises_90d"] = int((u.get("priceTargetAction", pd.Series(dtype=str)) == "Raises").sum())
        s["recent_actions"] = [
            f"{d.date()} {r.get('Firm','')}: {r.get('FromGrade','')}→{r.get('ToGrade','')}"
            f" PT {r.get('priorPriceTarget','')}→{r.get('currentPriceTarget','')}"
            for d, r in u.head(8).iterrows()]

    # B. 三方增长对比
    s["hist_rev_cagr_3y"] = hist_cagr(inc, ["Total Revenue", "Operating Revenue"])
    fcf = _ttm(qcf, ["Free Cash Flow"])
    s["fcf_ttm"] = fcf
    s["implied_fcf_growth"] = reverse_dcf(fcf, s["market_cap"], s["net_debt"]) if fcf else None
    s["signals"] = interpret(s)
    return s


def _ttm(qdf, names):
    if qdf is None or qdf.empty:
        return None
    for n in names:
        if n in qdf.index:
            v = qdf.loc[n].dropna()
            if len(v) >= 4:
                return float(v.iloc[:4].sum())
    return None


def hist_cagr(inc, names, years=3):
    if inc is None or inc.empty:
        return None
    for n in names:
        if n in inc.index:
            v = inc.loc[n].dropna()
            if len(v) > years and v.iloc[years] > 0 and v.iloc[0] > 0:
                return float((v.iloc[0] / v.iloc[years]) ** (1 / years) - 1)
    return None


def reverse_dcf(fcf, mcap, net_debt, r=CFG["discount_rate"], tg=CFG["terminal_growth"], n=CFG["dcf_years"]):
    """求解: 当前企业价值需要未来 n 年 FCF 年增长多少才能被证明合理。"""
    if not fcf or fcf <= 0 or not mcap:
        return None
    ev = mcap + (net_debt or 0)

    def pv(g):
        flows = sum(fcf * (1 + g) ** k / (1 + r) ** k for k in range(1, n + 1))
        term = fcf * (1 + g) ** n * (1 + tg) / (r - tg) / (1 + r) ** n
        return flows + term

    lo, hi = -0.5, 1.0
    if pv(lo) > ev:
        return lo
    if pv(hi) < ev:
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pv(mid) < ev else (lo, mid)
    return (lo + hi) / 2


def interpret(s):
    """把数字翻译成可读信号；这些是提示，不是结论。"""
    out = []
    g_mkt, g_ana, g_hist = s.get("implied_fcf_growth"), s.get("rev_growth_est_+1y"), s.get("hist_rev_cagr_3y")
    if g_mkt is not None and g_ana is not None:
        gap = g_ana - g_mkt
        if gap > 0.05:
            out.append(f"市场隐含增长({g_mkt:.0%})明显低于分析师预期({g_ana:.0%})：要么市场过度悲观，要么分析师还没下调")
        elif gap < -0.05:
            out.append(f"市场隐含增长({g_mkt:.0%})高于分析师预期({g_ana:.0%})：股价并不便宜")
    if g_hist is not None and g_ana is not None and g_ana < g_hist - 0.08:
        out.append(f"分析师预期增长({g_ana:.0%})远低于历史({g_hist:.0%})：需判断是周期性放缓还是结构性")
    r90 = s.get("eps_rev90_+1y")
    if r90 is not None and r90 < -0.05:
        out.append(f"明年EPS预期90天下修{r90:.0%}：分析师在跟随恶化，下修通常有惯性")
    elif r90 is not None and r90 > 0.02:
        out.append(f"股价下跌但明年EPS预期上修{r90:.0%}：基本面与价格背离，是最值得研究的情形")
    d = s.get("eps_dispersion_+1y")
    if d is not None and d > 0.4:
        out.append(f"分析师分歧大(高低差为均值的{d:.0%})：不确定性高，也是错误定价最可能出现的地方")
    if (s.get("pt_cuts_90d") or 0) >= 3 and (s.get("downgrades_90d") or 0) == 0 and (r90 or 0) > -0.03:
        out.append("目标价频繁下调但评级与EPS预期基本不变：可能只是目标价在追价格，而非新的基本面信息")
    br, su = s.get("beat_rate_4q"), s.get("avg_surprise_4q")
    if br is not None and br >= 0.75 and (su or 0) > 0.03:
        out.append(f"过去4季超预期率{br:.0%}、平均超{su:.0%}：分析师对这家公司历来偏保守")
    return out


# ============================== C. AI 审计员 ==============================
SYSTEM_PROMPT = """你是一名独立的卖方研究审计员。你的任务是审查华尔街一致预期，找出分析师可能判断失误的地方——但你必须保持严格的证据标准。

原则:
1. 默认立场是"市场和分析师大致正确"。只有找到具体、可验证、近期的证据时，才能判断分析师偏乐观或偏悲观。
2. 先搞清楚股价为什么跌（用搜索查近期新闻、财报、电话会、行业数据），再评价分析师。
3. 把一致预期拆成 3-5 个关键假设（如营收增速、利润率、资本开支、竞争格局、监管），逐条评估。
4. 寻找分析师的已知偏差：羊群效应、修正滞后、目标价追随股价、对周期拐点反应过度/不足。但不要把"可能有偏差"当成证据。
5. 同样认真地寻找空头论点。如果空头论点更有说服力，直接说分析师可能仍然偏乐观。
6. 每条判断都要附信息来源和日期。区分事实与推断。
7. "证据不足"是完全可以接受的结论，比勉强给出观点更好。

只输出一个 JSON 对象，不要任何其他文字或 markdown 代码块，结构如下:
{
  "why_fell": "股价下跌的主要原因，2-3句",
  "consensus_view": "分析师一致预期的核心逻辑，1-2句",
  "key_assumptions": [
    {"assumption": "...", "analyst_view": "...", "ai_assessment": "...",
     "direction": "analysts_too_pessimistic | analysts_too_optimistic | reasonable | unclear",
     "evidence": "具体证据及来源日期", "confidence": 0.0}
  ],
  "bear_case": "最强的空头论点",
  "bull_case": "最强的多头论点",
  "verdict": "possible_mispricing | analysts_likely_right | analysts_still_too_optimistic | insufficient_evidence",
  "verdict_confidence": 0.0,
  "one_line": "一句话结论（中文）",
  "what_would_prove_this_wrong": ["..."],
  "upcoming_catalysts": ["日期 + 事件"],
  "sources": ["url"]
}"""


def ai_audit(snap, cfg=CFG):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=cfg["model"], max_tokens=6000, system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": cfg["max_searches"]}],
        messages=[{"role": "user", "content": build_user_msg(snap)}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return parse_json(text)


def build_user_msg(snap):
    payload = {k: v for k, v in snap.items() if v is not None}
    return (f"今天是 {dt.date.today().isoformat()}。请审计 {snap['name']} ({snap['ticker']}) 的分析师一致预期。\n"
            f"以下是我的量化系统已经计算出的数据（来自 Yahoo Finance，可能有误差，请自行核实关键数字）:\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str, indent=1)}\n\n"
            f"说明: implied_fcf_growth 是反向DCF得出的、当前股价隐含的未来10年FCF年增长率；"
            f"eps_rev90 是一致预期90天变化；signals 是规则引擎生成的提示，不一定正确。")


# ============================== 免费模式：生成提示词 / 导入回复 ==============================
def snap_dir(cfg=CFG):
    d = Path(cfg["out_dir"]) / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reply_dir(cfg=CFG):
    d = Path(cfg["out_dir"]) / "replies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_prompts(tickers, cfg=CFG):
    parts = [f"# 分析师预期审计提示词 · {dt.date.today()}\n\n"
             f"用法：每次复制一段（从 ===== 开始到下一段之前）贴进一个新对话，务必打开联网搜索。\n"
             f"把 AI 回复的 JSON 原样保存为 `{reply_dir(cfg)}/代码.json`，全部完成后运行 "
             f"`python analyst_ai.py --import`。\n"]
    for tk in tickers:
        print(f"→ {tk} 拉取分析师数据…")
        snap = analyst_snapshot(tk)
        (snap_dir(cfg) / f"{tk}.json").write_text(
            json.dumps(snap, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
        parts.append(f"\n\n===== {tk} =====\n\n"
                     f"请严格按以下角色和规则工作，并使用联网搜索查找最新信息。\n\n"
                     f"{SYSTEM_PROMPT}\n\n---\n\n{build_user_msg(snap)}\n")
    f = Path(cfg["out_dir"]) / f"prompts_{dt.date.today().isoformat()}.md"
    f.write_text("".join(parts), encoding="utf-8")
    print(f"\n提示词已生成: {f}")


def import_replies(cfg=CFG):
    pairs = []
    for rf in sorted(reply_dir(cfg).glob("*.json")) + sorted(reply_dir(cfg).glob("*.txt")):
        tk = rf.stem.upper()
        sf = snap_dir(cfg) / f"{tk}.json"
        if not sf.exists():
            print(f"跳过 {tk}: 没有对应的数据快照（请先用 --prompt 生成）")
            continue
        snap = json.loads(sf.read_text(encoding="utf-8"))
        audit = parse_json(rf.read_text(encoding="utf-8"))
        log_verdict(snap, audit)
        pairs.append((snap, audit))
        done = reply_dir(cfg) / "imported"
        done.mkdir(exist_ok=True)
        rf.rename(done / f"{dt.date.today().isoformat()}_{rf.name}")
        print(f"已导入 {tk}: {audit.get('one_line', '')}")
    if pairs:
        print(f"\n报告: {write_report(pairs, cfg)}")
    else:
        print("replies/ 文件夹里没有可导入的回复。")


def parse_json(text):
    t = text.strip().replace("```json", "").replace("```", "").strip()
    a, b = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[a:b + 1])
    except Exception:
        return {"verdict": "parse_error", "one_line": "AI 输出无法解析", "raw": text[:3000]}


# ============================== D. 记账与回测 ==============================
def log_verdict(snap, audit, path=CFG["ledger"]):
    row = {"date": dt.date.today().isoformat(), "ticker": snap["ticker"], "price": snap.get("price"),
           "verdict": audit.get("verdict"), "confidence": audit.get("verdict_confidence"),
           "implied_growth": snap.get("implied_fcf_growth"), "analyst_growth": snap.get("rev_growth_est_+1y"),
           "eps_rev90": snap.get("eps_rev90_+1y"), "one_line": audit.get("one_line")}
    p = Path(path)
    p.parent.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)


def evaluate(path=CFG["ledger"]):
    """对每条历史判断计算之后 63/126 个交易日相对 SPY 的超额收益，按 verdict 汇总。"""
    led = pd.read_csv(path, parse_dates=["date"])
    tickers = sorted(set(led["ticker"])) + ["SPY"]
    px = yf.download(tickers, start=led["date"].min() - pd.Timedelta(days=5),
                     auto_adjust=True, progress=False)["Close"]
    def fwd(tk, d, n):
        s = px[tk].dropna()
        s0 = s[s.index >= d]
        return s0.iloc[n] / s0.iloc[0] - 1 if len(s0) > n else np.nan
    for n, lab in ((63, "3m"), (126, "6m")):
        led[f"excess_{lab}"] = [fwd(r.ticker, r.date, n) - fwd("SPY", r.date, n) for r in led.itertuples()]
    summary = led.groupby("verdict")[["excess_3m", "excess_6m"]].agg(["mean", "count"])
    print(summary.round(3))
    print("\n若 possible_mispricing 组没有显著跑赢 analysts_likely_right 组，说明 AI 的'反向判断'目前没有价值。")
    return led


# ============================== 报告 ==============================
def pct(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


VERDICT_CN = {"possible_mispricing": ("可能错杀", "#067647"),
              "analysts_likely_right": ("分析师大概率正确", "#475467"),
              "analysts_still_too_optimistic": ("分析师可能仍偏乐观", "#b42318"),
              "insufficient_evidence": ("证据不足", "#98a2b3"),
              "parse_error": ("解析失败", "#98a2b3")}


def card(s, a):
    vcn, color = VERDICT_CN.get(a.get("verdict"), (a.get("verdict"), "#475467"))
    rows = [("当前股价隐含FCF增长", pct(s.get("implied_fcf_growth"))),
            ("分析师明年营收增长", pct(s.get("rev_growth_est_+1y"))),
            ("历史3年营收CAGR", pct(s.get("hist_rev_cagr_3y"))),
            ("明年EPS预期 30天/90天变化", f"{pct(s.get('eps_rev30_+1y'))} / {pct(s.get('eps_rev90_+1y'))}"),
            ("EPS分歧度(高-低)/均值", pct(s.get("eps_dispersion_+1y"))),
            ("过去4季超预期率 / 平均幅度", f"{pct(s.get('beat_rate_4q'))} / {pct(s.get('avg_surprise_4q'))}"),
            ("平均目标价空间", pct(s.get("pt_upside"))),
            ("买入评级占比", pct(s.get("pct_buy"))),
            ("90天 升级/降级 · 目标价上调/下调",
             f"{s.get('upgrades_90d','—')}/{s.get('downgrades_90d','—')} · {s.get('pt_raises_90d','—')}/{s.get('pt_cuts_90d','—')}")]
    tbl = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    sig = "".join(f"<li>{x}</li>" for x in s.get("signals", []))
    ass = "".join(
        f"<tr><td>{x.get('assumption','')}</td><td>{x.get('analyst_view','')}</td>"
        f"<td>{x.get('ai_assessment','')}<br><small>{x.get('evidence','')}</small></td>"
        f"<td>{x.get('direction','')}<br>{x.get('confidence','')}</td></tr>"
        for x in a.get("key_assumptions", []))
    lst = lambda k: "".join(f"<li>{x}</li>" for x in a.get(k, []))
    src = "".join(f"<li><a href='{u}'>{u}</a></li>" for u in a.get("sources", []))
    return f"""
<section class=card>
 <h2>{s['name']} <span class=tk>{s['ticker']}</span>
   <span class=v style="background:{color}">{vcn} · {a.get('verdict_confidence','')}</span></h2>
 <p class=one>{a.get('one_line','')}</p>
 <div class=grid><div><h3>分析师数据</h3><table>{tbl}</table>
   <h3>规则引擎提示</h3><ul>{sig or '<li>无</li>'}</ul></div>
  <div><h3>为什么跌</h3><p>{a.get('why_fell','')}</p>
   <h3>一致预期逻辑</h3><p>{a.get('consensus_view','')}</p>
   <h3>多头 / 空头</h3><p><b>多：</b>{a.get('bull_case','')}</p><p><b>空：</b>{a.get('bear_case','')}</p></div></div>
 <h3>关键假设逐条审计</h3>
 <table class=ass><tr><th>假设</th><th>分析师</th><th>AI评估与证据</th><th>方向/置信</th></tr>{ass}</table>
 <div class=grid><div><h3>什么会证明这个判断错了</h3><ul>{lst('what_would_prove_this_wrong')}</ul></div>
  <div><h3>近期催化剂</h3><ul>{lst('upcoming_catalysts')}</ul></div></div>
 <details><summary>来源</summary><ul>{src}</ul></details>
</section>"""


CSS = """<style>
body{font-family:-apple-system,"PingFang SC",Segoe UI,sans-serif;margin:0;padding:32px;background:#f6f6f4;color:#1a1a1a}
.card{background:#fff;border-radius:10px;padding:24px 28px;margin:0 auto 24px;max-width:1100px}
h2{margin:0 0 6px;font-size:20px}.tk{color:#888;font-weight:400}
.v{color:#fff;font-size:12px;padding:3px 8px;border-radius:4px;margin-left:8px;vertical-align:middle}
.one{font-size:15px;margin:4px 0 16px}h3{font-size:13px;color:#555;margin:16px 0 6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{padding:5px 6px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
.ass td:nth-child(3){width:45%}small{color:#777}ul{padding-left:18px;font-size:13px}p{font-size:13.5px;line-height:1.55}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style>"""


def write_report(pairs, cfg=CFG):
    out = Path(cfg["out_dir"])
    out.mkdir(exist_ok=True)
    f = out / f"analyst_audit_{dt.date.today().isoformat()}.html"
    body = "".join(card(s, a) for s, a in pairs)
    f.write_text(f"<html><head><meta charset=utf-8>{CSS}</head><body>"
                 f"<div class=card><h2>分析师预期审计 · {dt.date.today()}</h2>"
                 f"<p>AI 判断仅供研究参考，可能出错。请结合 ledger.csv 的历史命中率评估其可信度。</p></div>"
                 f"{body}</body></html>", encoding="utf-8")
    return f


# ============================== 主流程 ==============================
def pick_tickers(argv, cfg=CFG):
    if argv:
        return [a.upper() for a in argv]
    files = sorted(glob.glob(os.path.join(cfg["out_dir"], "screen_*.csv")))
    if not files:
        sys.exit("没有找到 screen_*.csv，请先运行 oversold_quality_screener.py 或直接指定股票代码。")
    return pd.read_csv(files[-1])["ticker"].head(cfg["top_n"]).tolist()


def main():
    args = sys.argv[1:]
    if args and args[0] == "--evaluate":
        evaluate()
        return
    if args and args[0] == "--prompt":
        make_prompts(pick_tickers(args[1:]))
        return
    if args and args[0] == "--import":
        import_replies()
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("未设置 ANTHROPIC_API_KEY，自动切换到免费模式（生成提示词）。")
        make_prompts(pick_tickers(args))
        return
    pairs = []
    for tk in pick_tickers(args):
        print(f"→ {tk} 拉取分析师数据…")
        snap = analyst_snapshot(tk)
        print("   AI 审计中（会联网搜索，约 1-2 分钟）…")
        try:
            audit = ai_audit(snap)
        except Exception as e:
            audit = {"verdict": "parse_error", "one_line": f"调用失败: {e}"}
        log_verdict(snap, audit)
        print(f"   {VERDICT_CN.get(audit.get('verdict'), (audit.get('verdict'),))[0]}: {audit.get('one_line','')}")
        pairs.append((snap, audit))
    print(f"\n报告: {write_report(pairs)}")


if __name__ == "__main__":
    main()
