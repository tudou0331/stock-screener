"""
oversold_quality_screener.py  —  超卖 + 基本面健康 股票筛选器 v1

流水线:
  1. 股票池        S&P 500 / Nasdaq 100 / 自选
  2. 技术面超卖    批量下载价格，算 RSI、距52周高点回撤、200日均线偏离、放量
  3. 基本面硬门槛  只对超卖候选逐只拉财报：FCF、杠杆、利息覆盖、毛利、ROE、营收增长、Piotroski F-Score
  4. 价值陷阱预警  分析师EPS预期下修、相对板块的个股特有下跌、临近财报、最新新闻标题
  5. 综合排序      输出 CSV + HTML 报告

安装:  pip install yfinance pandas numpy requests lxml jinja2 matplotlib
运行:  python oversold_quality_screener.py
"""
import io
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================== 配置 ==============================
CONFIG = {
    "universe": ["SP500", "NDX100"],          # 可选 "SP500", "NDX100"
    "extra_tickers": [],                      # 自选；欧股加交易所后缀，如 "MC.PA", "ASML.AS"
    "exclude_sectors": ["Financial Services", "Real Estate"],  # 杠杆/EBITDA 指标对银行、REIT 不适用
    "min_market_cap": 5e9,

    # --- 技术面：超卖定义 ---
    "rsi_max": 35,                 # RSI(14) 上限
    "drawdown_min": 0.20,          # 距 52 周高点至少回撤 20%
    "require_below_200dma": False, # 是否强制要求跌破 200 日均线

    # --- 基本面：硬门槛（任一不满足即淘汰）---
    "min_fcf_margin": 0.05,        # TTM 自由现金流率
    "max_net_debt_ebitda": 2.5,    # 净负债 / EBITDA
    "min_interest_coverage": 6.0,  # EBIT / 利息支出
    "min_gross_margin": 0.25,
    "min_roe": 0.10,               # 股东权益为负（大额回购）时跳过此项
    "min_revenue_growth": 0.0,     # 最近季度营收同比
    "min_piotroski": 6,            # F-Score 0-9

    # --- 预警阈值 ---
    "eps_cut_flag": -0.08,         # 当年/明年 EPS 一致预期 90 天内下修超过 8% 则预警
    "idio_drop_flag": -0.10,       # 21 日跑输板块 ETF 超过 10% 则预警
    "earnings_days_flag": 14,

    "price_period": "2y",
    "sleep_between_calls": 0.3,
    "output_dir": "reports",
}

SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Communication Services": "XLC", "Industrials": "XLI",
    "Energy": "XLE", "Basic Materials": "XLB", "Utilities": "XLU",
    "Financial Services": "XLF", "Real Estate": "XLRE",
}

UA = {"User-Agent": "Mozilla/5.0 (screener)"}


# ============================== 1. 股票池 ==============================
def _wiki_tickers(url):
    html = requests.get(url, headers=UA, timeout=30).text
    for tbl in pd.read_html(io.StringIO(html)):
        for col in ("Symbol", "Ticker"):
            if col in tbl.columns and len(tbl) > 50:
                return tbl[col].astype(str).str.replace(".", "-", regex=False).tolist()
    return []


def build_universe(cfg):
    tickers = []
    if "SP500" in cfg["universe"]:
        tickers += _wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    if "NDX100" in cfg["universe"]:
        tickers += _wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100")
    tickers += cfg["extra_tickers"]
    return sorted(set(t.strip().upper() for t in tickers if t.strip()))


# ============================== 2. 技术面 ==============================
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def download_prices(tickers, period, chunk=150):
    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        df = yf.download(part, period=period, auto_adjust=True, group_by="column",
                         threads=True, progress=False)
        if df.empty:
            continue
        c, v = df["Close"], df["Volume"]
        if isinstance(c, pd.Series):  # 单只股票时
            c, v = c.to_frame(part[0]), v.to_frame(part[0])
        closes.append(c)
        vols.append(v)
    return pd.concat(closes, axis=1), pd.concat(vols, axis=1)


def technical_table(close, vol):
    rows = []
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 210:
            continue
        v = vol[t].reindex(c.index).fillna(0)
        last = c.iloc[-1]
        hi52 = c.iloc[-252:].max()
        ma200 = c.iloc[-200:].mean()
        rows.append({
            "ticker": t,
            "price": last,
            "rsi14": rsi(c).iloc[-1],
            "drawdown": 1 - last / hi52,
            "vs_200dma": last / ma200 - 1,
            "ret_21d": last / c.iloc[-22] - 1,
            "vol_ratio": v.iloc[-5:].mean() / max(v.iloc[-55:-5].mean(), 1),
        })
    return pd.DataFrame(rows).set_index("ticker")


def oversold_filter(tech, cfg):
    m = (tech["rsi14"] <= cfg["rsi_max"]) & (tech["drawdown"] >= cfg["drawdown_min"])
    if cfg["require_below_200dma"]:
        m &= tech["vs_200dma"] < 0
    out = tech[m].copy()
    # 超卖强度：RSI 越低、回撤越深、越放量 → 分越高
    out["oversold_score"] = ((cfg["rsi_max"] - out["rsi14"]) / cfg["rsi_max"]
                             + out["drawdown"]
                             + 0.1 * np.log(out["vol_ratio"].clip(0.5, 5)))
    return out


# ============================== 3. 基本面 ==============================
def _row(df, names):
    """在财报 DataFrame 中按候选行名取一行（列按日期倒序）。"""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if len(s):
                return s
    return None


def _ttm(qdf, names):
    s = _row(qdf, names)
    return float(s.iloc[:4].sum()) if s is not None and len(s) >= 4 else None


def piotroski(inc, bs, cf):
    """标准 9 项 F-Score，用最近两个年报。返回 (score, 可计算项数)。"""
    def two(df, names):
        s = _row(df, names)
        return (float(s.iloc[0]), float(s.iloc[1])) if s is not None and len(s) >= 2 else (None, None)

    ni, ni_p = two(inc, ["Net Income", "Net Income Common Stockholders"])
    ta, ta_p = two(bs, ["Total Assets"])
    cfo, _ = two(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    ltd, ltd_p = two(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
    ca, ca_p = two(bs, ["Current Assets"])
    cl, cl_p = two(bs, ["Current Liabilities"])
    sh, sh_p = two(bs, ["Ordinary Shares Number", "Share Issued"])
    gp, gp_p = two(inc, ["Gross Profit"])
    rev, rev_p = two(inc, ["Total Revenue", "Operating Revenue"])

    tests = []
    def add(cond_fn, *vals):
        if all(x is not None for x in vals):
            try:
                tests.append(bool(cond_fn()))
            except ZeroDivisionError:
                pass

    add(lambda: ni / ta > 0, ni, ta)
    add(lambda: cfo > 0, cfo)
    add(lambda: ni / ta > ni_p / ta_p, ni, ta, ni_p, ta_p)
    add(lambda: cfo > ni, cfo, ni)
    add(lambda: ltd / ta <= ltd_p / ta_p, ltd, ta, ltd_p, ta_p)
    add(lambda: ca / cl > ca_p / cl_p, ca, cl, ca_p, cl_p)
    add(lambda: sh <= sh_p * 1.01, sh, sh_p)
    add(lambda: gp / rev > gp_p / rev_p, gp, rev, gp_p, rev_p)
    add(lambda: rev / ta > rev_p / ta_p, rev, ta, rev_p, ta_p)
    return sum(tests), len(tests)


def fundamentals(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    inc, bs, cf = t.financials, t.balance_sheet, t.cashflow
    qinc, qcf = t.quarterly_financials, t.quarterly_cashflow

    rev_ttm = _ttm(qinc, ["Total Revenue", "Operating Revenue"]) or info.get("totalRevenue")
    fcf_ttm = _ttm(qcf, ["Free Cash Flow"])
    if fcf_ttm is None:
        cfo = _ttm(qcf, ["Operating Cash Flow"])
        capex = _ttm(qcf, ["Capital Expenditure"])
        fcf_ttm = cfo + capex if cfo is not None and capex is not None else info.get("freeCashflow")

    ebit = _row(inc, ["EBIT", "Operating Income"])
    intexp = _row(inc, ["Interest Expense", "Interest Expense Non Operating"])
    if ebit is not None and intexp is not None and abs(intexp.iloc[0]) > 0:
        cover = float(ebit.iloc[0]) / abs(float(intexp.iloc[0]))
    else:
        cover = np.inf  # 无利息支出 ≈ 无有息负债

    debt, cash, ebitda = info.get("totalDebt") or 0, info.get("totalCash") or 0, info.get("ebitda")
    nd_ebitda = (debt - cash) / ebitda if ebitda and ebitda > 0 else np.nan
    mcap = info.get("marketCap")
    f_score, f_n = piotroski(inc, bs, cf)

    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector"),
        "market_cap": mcap,
        "fcf_margin": fcf_ttm / rev_ttm if fcf_ttm is not None and rev_ttm else np.nan,
        "fcf_yield": fcf_ttm / mcap if fcf_ttm is not None and mcap else np.nan,
        "net_debt_ebitda": nd_ebitda,
        "interest_cover": cover,
        "gross_margin": info.get("grossMargins"),
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "fwd_pe": info.get("forwardPE"),
        "piotroski": f_score,
        "piotroski_n": f_n,
        "_ticker_obj": t,
    }


def quality_gate(f, cfg):
    """返回未通过的门槛列表；空列表 = 通过。缺数据视为不通过（宁缺毋滥）。"""
    fails = []
    def chk(name, val, ok):
        if val is None or (isinstance(val, float) and np.isnan(val)) or not ok(val):
            fails.append(name)
    chk("市值", f["market_cap"], lambda x: x >= cfg["min_market_cap"])
    chk("FCF率", f["fcf_margin"], lambda x: x >= cfg["min_fcf_margin"])
    nd = f["net_debt_ebitda"]
    if not (isinstance(nd, float) and np.isnan(nd)):
        chk("净负债/EBITDA", nd, lambda x: x <= cfg["max_net_debt_ebitda"])
    else:
        fails.append("EBITDA≤0")
    chk("利息覆盖", f["interest_cover"], lambda x: x >= cfg["min_interest_coverage"])
    chk("毛利率", f["gross_margin"], lambda x: x >= cfg["min_gross_margin"])
    if f["roe"] is not None:  # 负权益公司 yfinance 通常返回 None，跳过
        chk("ROE", f["roe"], lambda x: x >= cfg["min_roe"])
    chk("营收增长", f["rev_growth"], lambda x: x >= cfg["min_revenue_growth"])
    if f["piotroski_n"] >= 7:
        chk("F-Score", f["piotroski"], lambda x: x >= cfg["min_piotroski"])
    else:
        fails.append("F-Score数据不足")
    return fails


# ============================== 4. 价值陷阱预警 ==============================
def red_flags(t, ticker, tech_row, sector, close, cfg):
    flags, notes = [], {}

    # 4a. 分析师 EPS 一致预期 90 天变化 —— 最重要的价值陷阱信号
    try:
        tr = t.eps_trend
        for per in ("0y", "+1y"):
            if tr is not None and per in tr.index:
                cur, old = tr.loc[per, "current"], tr.loc[per, "90daysAgo"]
                if old and old > 0:
                    chg = cur / old - 1
                    notes[f"eps_rev_{per}"] = chg
                    if chg <= cfg["eps_cut_flag"]:
                        flags.append(f"EPS预期{per}下修{chg:.0%}")
    except Exception:
        pass

    # 4b. 个股特有下跌 vs 板块
    etf = SECTOR_ETF.get(sector)
    if etf and etf in close.columns:
        e = close[etf].dropna()
        rel = tech_row["ret_21d"] - (e.iloc[-1] / e.iloc[-22] - 1)
        notes["rel_sector_21d"] = rel
        if rel <= cfg["idio_drop_flag"]:
            flags.append(f"跑输{etf} {rel:.0%}（个股原因）")

    # 4c. 临近财报
    try:
        cal = t.calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d = pd.Timestamp(dates[0]).date()
            days = (d - dt.date.today()).days
            notes["next_earnings"] = d.isoformat()
            if 0 <= days <= cfg["earnings_days_flag"]:
                flags.append(f"{days}天后财报")
    except Exception:
        pass

    # 4d. 最新新闻标题（供人工判断下跌原因）
    try:
        heads = []
        for n in (t.news or [])[:3]:
            title = n.get("title") or (n.get("content") or {}).get("title")
            if title:
                heads.append(title)
        notes["headlines"] = " | ".join(heads)
    except Exception:
        notes["headlines"] = ""
    return flags, notes


# ============================== 5. 主流程 ==============================
def rank_pct(s, ascending=True):
    return s.rank(pct=True, ascending=ascending).fillna(0)


def run(cfg=CONFIG):
    t0 = time.time()
    universe = build_universe(cfg)
    print(f"[1] 股票池 {len(universe)} 只")

    close, vol = download_prices(universe + list(SECTOR_ETF.values()) + ["SPY"], cfg["price_period"])
    tech = technical_table(close[[c for c in universe if c in close.columns]],
                           vol[[c for c in universe if c in vol.columns]])
    cands = oversold_filter(tech, cfg)
    print(f"[2] 技术面超卖 {len(cands)} 只")

    passed, rejected = [], []
    for i, tk in enumerate(cands.index, 1):
        try:
            f = fundamentals(tk)
        except Exception as e:
            rejected.append({"ticker": tk, "fails": f"数据错误: {e}"})
            continue
        if f["sector"] in cfg["exclude_sectors"]:
            continue
        fails = quality_gate(f, cfg)
        if fails:
            rejected.append({"ticker": tk, "name": f["name"], "fails": ", ".join(fails)})
        else:
            flags, notes = red_flags(f.pop("_ticker_obj"), tk, cands.loc[tk], f["sector"], close, cfg)
            passed.append({"ticker": tk, **f, **cands.loc[tk].to_dict(), **notes,
                           "flags": "; ".join(flags), "n_flags": len(flags)})
        time.sleep(cfg["sleep_between_calls"])
        print(f"    {i}/{len(cands)} {tk}: {'通过' if not fails else '淘汰'}", end="\r")
    print()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)
    stamp = dt.date.today().isoformat()

    if not passed:
        print("[3] 今天没有同时满足超卖 + 质量门槛的股票。")
        pd.DataFrame(rejected).to_csv(out_dir / f"rejected_{stamp}.csv", index=False)
        return None

    df = pd.DataFrame(passed).set_index("ticker")
    df["quality_score"] = (df["piotroski"] / 9 + df["fcf_margin"].clip(0, 0.4)
                           + df["gross_margin"].clip(0, 0.8) / 2)
    df["score"] = (0.35 * rank_pct(df["oversold_score"])
                   + 0.35 * rank_pct(df["quality_score"])
                   + 0.30 * rank_pct(df["fcf_yield"])
                   - 0.10 * df["n_flags"])
    df = df.sort_values("score", ascending=False)
    print(f"[3] 通过质量门槛 {len(df)} 只，用时 {time.time() - t0:.0f}s")

    cols = ["name", "sector", "price", "score", "rsi14", "drawdown", "vs_200dma",
            "fcf_yield", "fcf_margin", "net_debt_ebitda", "interest_cover", "gross_margin",
            "roe", "rev_growth", "piotroski", "fwd_pe", "eps_rev_0y", "eps_rev_+1y",
            "rel_sector_21d", "next_earnings", "flags", "headlines"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(out_dir / f"screen_{stamp}.csv")
    pd.DataFrame(rejected).to_csv(out_dir / f"rejected_{stamp}.csv", index=False)
    write_html(df, out_dir / f"screen_{stamp}.html", stamp)
    print(df[["name", "score", "rsi14", "drawdown", "fcf_yield", "flags"]].head(15).to_string())
    return df


def write_html(df, path, stamp):
    pct = ["drawdown", "vs_200dma", "fcf_yield", "fcf_margin", "gross_margin", "roe",
           "rev_growth", "eps_rev_0y", "eps_rev_+1y", "rel_sector_21d"]
    fmt = {c: "{:.1%}" for c in pct if c in df.columns}
    fmt.update({c: "{:.2f}" for c in ["price", "score", "rsi14", "net_debt_ebitda",
                                       "interest_cover", "fwd_pe"] if c in df.columns})
    styled = (df.style.format(fmt, na_rep="—")
              .background_gradient(subset=["score"], cmap="Greens")
              .map(lambda v: "color:#b42318;font-weight:600" if isinstance(v, str) and v else "",
                   subset=["flags"]))
    css = ("<style>body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;color:#1a1a1a}"
           "table{border-collapse:collapse;font-size:12px}th,td{padding:6px 8px;border-bottom:1px solid #eee;"
           "text-align:right;white-space:nowrap}td:last-child{white-space:normal;max-width:420px;text-align:left}"
           "th{background:#fafafa;position:sticky;top:0}</style>")
    Path(path).write_text(f"<html><head><meta charset='utf-8'>{css}</head><body>"
                          f"<h2>超卖 + 质量筛选 · {stamp}</h2>"
                          f"<p>候选清单，不构成投资建议。红色预警项需人工核实下跌原因。</p>"
                          f"{styled.to_html()}</body></html>", encoding="utf-8")


if __name__ == "__main__":
    run()
