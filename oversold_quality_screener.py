"""
oversold_quality_screener.py  —  超卖 + 基本面健康 股票筛选器 v2（双通道）

流水线:
  1. 股票池      全美上市股票（纳斯达克官方列表，按市值和成交额过滤）；失败时退回 S&P 500/400/600 + Nasdaq 100
  2. 技术面超卖  批量下载价格：RSI、距52周高点回撤、200日均线偏离、放量
  3. 双通道基本面
       价值通道：已盈利的成熟公司 —— FCF、杠杆、利息覆盖、毛利、ROE、营收增长、Piotroski
       成长通道：高增长、可能尚未盈利 —— 营收增速、毛利率、40法则、现金跑道、稀释速度、利润率改善
  4. 价值陷阱预警  EPS预期下修、相对板块的个股特有下跌、临近财报、新闻标题
  5. 通道内排序   输出 CSV + HTML 报告（两个通道分开显示）

安装:  pip install yfinance pandas numpy requests lxml jinja2 matplotlib
运行:  python oversold_quality_screener.py
"""
import io
import re
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================== 配置 ==============================
CONFIG = {
    # --- 股票池 ---
    "universe_source": "nasdaq_all",   # "nasdaq_all" = 全美上市股票；"indices" = 只用下面的指数
    "indices": ["SP500", "SP400", "SP600", "NDX100"],
    "extra_tickers": [],               # 自选；欧股加后缀，如 "MC.PA", "ASML.AS"
    "min_universe_cap": 1e9,           # 进入股票池的最低市值
    "min_dollar_volume": 5e6,          # 近 50 日平均每日成交额，过滤流动性差的股票
    "exclude_sectors": ["Financial Services", "Real Estate"],

    # --- 价值通道：已盈利、财务稳健 ---
    "value": {
        "enabled": True,
        "min_market_cap": 2e9,
        "rsi_max": 35,
        "drawdown_min": 0.20,
        "min_fcf_margin": 0.05,
        "max_net_debt_ebitda": 2.5,
        "min_interest_coverage": 6.0,
        "min_gross_margin": 0.25,
        "min_roe": 0.10,
        "min_revenue_growth": 0.0,
        "min_piotroski": 6,
    },

    # --- 成长通道：高增长，允许尚未盈利 ---
    "growth": {
        "enabled": True,
        "min_market_cap": 1e9,
        "rsi_max": 35,
        "drawdown_min": 0.35,          # 成长股波动大，回撤门槛更高
        "min_revenue_growth": 0.20,    # TTM 营收同比
        "min_gross_margin": 0.40,
        "min_rule_of_40": 0.40,        # 营收增速 + FCF率
        "min_runway_years": 2.0,       # 烧钱公司：现金 / 年烧钱额
        "max_share_dilution": 0.08,    # 年度股数增长上限
        "require_margin_improving": True,      # 经营利润率较上年改善
        "require_net_cash_if_burning": True,   # 烧钱公司必须现金 ≥ 负债
    },

    # --- 预警阈值 ---
    "eps_cut_flag": -0.08,
    "idio_drop_flag": -0.10,
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
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}


# ============================== 1. 股票池 ==============================
def parse_nasdaq_rows(rows, min_cap):
    out = []
    for r in rows:
        sym = str(r.get("symbol", "")).strip().upper()
        if not re.fullmatch(r"[A-Z]{1,5}(/[A-Z])?", sym):      # 排除优先股、权证、单位等
            continue
        if "Blank Check" in str(r.get("industry", "")):         # 排除 SPAC
            continue
        try:
            cap = float(str(r.get("marketCap", "0")).replace(",", "") or 0)
        except ValueError:
            cap = 0
        if cap >= min_cap:
            out.append(sym.replace("/", "-"))
    return out


def nasdaq_all(min_cap):
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
    js = requests.get(url, headers=UA, timeout=60).json()
    rows = (js.get("data") or {}).get("rows") or []
    return parse_nasdaq_rows(rows, min_cap)


def _wiki_tickers(url):
    html = requests.get(url, headers=UA, timeout=30).text
    for tbl in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in tbl.columns]
        for col in ("Symbol", "Ticker", "Ticker symbol"):
            if col in cols and len(tbl) > 50:
                return tbl[col].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
    return []


WIKI = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    "NDX100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


def build_universe(cfg):
    tickers = []
    if cfg["universe_source"] == "nasdaq_all":
        try:
            tickers = nasdaq_all(cfg["min_universe_cap"])
            print(f"    纳斯达克全市场列表: {len(tickers)} 只（市值 ≥ {cfg['min_universe_cap']/1e9:.0f}B）")
        except Exception as e:
            print(f"    纳斯达克列表获取失败（{e}），改用指数成分股")
    if len(tickers) < 500:
        for idx in cfg["indices"]:
            try:
                got = _wiki_tickers(WIKI[idx])
                print(f"    {idx}: {len(got)} 只")
                tickers += got
            except Exception as e:
                print(f"    {idx} 获取失败: {e}")
    tickers += cfg["extra_tickers"]
    return sorted(set(t.strip().upper() for t in tickers if t.strip()))


# ============================== 2. 技术面 ==============================
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def download_prices(tickers, period, chunk=200):
    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(2):
            try:
                df = yf.download(part, period=period, auto_adjust=True, group_by="column",
                                 threads=True, progress=False)
                break
            except Exception as e:
                print(f"    下载失败重试: {e}")
                time.sleep(5)
                df = pd.DataFrame()
        if df.empty:
            continue
        c, v = df["Close"], df["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(part[0]), v.to_frame(part[0])
        closes.append(c)
        vols.append(v)
        print(f"    价格下载 {min(i + chunk, len(tickers))}/{len(tickers)}", end="\r")
    print()
    return pd.concat(closes, axis=1), pd.concat(vols, axis=1)


def technical_table(close, vol, min_dollar_volume):
    rows = []
    for t in close.columns:
        c = close[t].dropna()
        if len(c) < 210:
            continue
        v = vol[t].reindex(c.index).fillna(0)
        dollar_vol = (c.iloc[-50:] * v.iloc[-50:]).mean()
        if dollar_vol < min_dollar_volume:
            continue
        last = c.iloc[-1]
        rows.append({
            "ticker": t,
            "price": last,
            "rsi14": rsi(c).iloc[-1],
            "drawdown": 1 - last / c.iloc[-252:].max(),
            "vs_200dma": last / c.iloc[-200:].mean() - 1,
            "ret_21d": last / c.iloc[-22] - 1,
            "vol_ratio": v.iloc[-5:].mean() / max(v.iloc[-55:-5].mean(), 1),
            "dollar_vol_m": dollar_vol / 1e6,
        })
    return pd.DataFrame(rows).set_index("ticker")


def oversold_filter(tech, cfg):
    tracks = [cfg[k] for k in ("value", "growth") if cfg[k]["enabled"]]
    rsi_max = max(t["rsi_max"] for t in tracks)
    dd_min = min(t["drawdown_min"] for t in tracks)
    out = tech[(tech["rsi14"] <= rsi_max) & (tech["drawdown"] >= dd_min)].copy()
    out["oversold_score"] = ((rsi_max - out["rsi14"]) / rsi_max + out["drawdown"]
                             + 0.1 * np.log(out["vol_ratio"].clip(0.5, 5)))
    return out


# ============================== 3. 基本面 ==============================
def _row(df, names):
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna()
            if len(s):
                return s
    return None


def _ttm(qdf, names, start=0):
    s = _row(qdf, names)
    return float(s.iloc[start:start + 4].sum()) if s is not None and len(s) >= start + 4 else None


def piotroski(inc, bs, cf):
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
    def add(fn, *vals):
        if all(x is not None for x in vals):
            try:
                tests.append(bool(fn()))
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
    REV = ["Total Revenue", "Operating Revenue"]

    rev_ttm = _ttm(qinc, REV) or info.get("totalRevenue")
    rev_prev = _ttm(qinc, REV, start=4)
    rev_growth_ttm = rev_ttm / rev_prev - 1 if rev_ttm and rev_prev and rev_prev > 0 else info.get("revenueGrowth")

    fcf_ttm = _ttm(qcf, ["Free Cash Flow"])
    if fcf_ttm is None:
        cfo, capex = _ttm(qcf, ["Operating Cash Flow"]), _ttm(qcf, ["Capital Expenditure"])
        fcf_ttm = cfo + capex if cfo is not None and capex is not None else info.get("freeCashflow")

    ebit = _row(inc, ["EBIT", "Operating Income"])
    intexp = _row(inc, ["Interest Expense", "Interest Expense Non Operating"])
    cover = (float(ebit.iloc[0]) / abs(float(intexp.iloc[0]))
             if ebit is not None and intexp is not None and abs(intexp.iloc[0]) > 0 else np.inf)

    op = _row(inc, ["Operating Income"])
    rv = _row(inc, REV)
    op_m = op_m_prev = None
    if op is not None and rv is not None and len(op) >= 2 and len(rv) >= 2 and rv.iloc[0] and rv.iloc[1]:
        op_m, op_m_prev = float(op.iloc[0] / rv.iloc[0]), float(op.iloc[1] / rv.iloc[1])

    sh = _row(bs, ["Ordinary Shares Number", "Share Issued"])
    share_growth = float(sh.iloc[0] / sh.iloc[1] - 1) if sh is not None and len(sh) >= 2 and sh.iloc[1] else None

    debt, cash, ebitda = info.get("totalDebt") or 0, info.get("totalCash") or 0, info.get("ebitda")
    nd_ebitda = (debt - cash) / ebitda if ebitda and ebitda > 0 else np.nan
    runway = cash / -fcf_ttm if fcf_ttm is not None and fcf_ttm < 0 else np.inf
    mcap = info.get("marketCap")
    fcf_margin = fcf_ttm / rev_ttm if fcf_ttm is not None and rev_ttm else np.nan
    f_score, f_n = piotroski(inc, bs, cf)

    return {
        "name": info.get("shortName", ticker),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": mcap,
        "fcf_margin": fcf_margin,
        "fcf_yield": fcf_ttm / mcap if fcf_ttm is not None and mcap else np.nan,
        "net_debt_ebitda": nd_ebitda,
        "interest_cover": cover,
        "gross_margin": info.get("grossMargins"),
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "rev_growth_ttm": rev_growth_ttm,
        "rule_of_40": (rev_growth_ttm or 0) + (fcf_margin if not np.isnan(fcf_margin) else -1),
        "op_margin": op_m,
        "op_margin_prev": op_m_prev,
        "share_growth": share_growth,
        "cash": cash,
        "debt": debt,
        "runway_years": runway,
        "ev_sales": info.get("enterpriseToRevenue"),
        "fwd_pe": info.get("forwardPE"),
        "piotroski": f_score,
        "piotroski_n": f_n,
        "_ticker_obj": t,
    }


def _bad(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def value_gate(f, tech, g):
    """返回未通过的门槛列表；空 = 通过。缺数据视为不通过。"""
    fails = []
    def chk(name, val, ok):
        if _bad(val) or not ok(val):
            fails.append(name)
    if tech["rsi14"] > g["rsi_max"] or tech["drawdown"] < g["drawdown_min"]:
        fails.append("未达价值通道超卖标准")
    chk("市值", f["market_cap"], lambda x: x >= g["min_market_cap"])
    chk("FCF率", f["fcf_margin"], lambda x: x >= g["min_fcf_margin"])
    if _bad(f["net_debt_ebitda"]):
        fails.append("EBITDA≤0")
    else:
        chk("净负债/EBITDA", f["net_debt_ebitda"], lambda x: x <= g["max_net_debt_ebitda"])
    chk("利息覆盖", f["interest_cover"], lambda x: x >= g["min_interest_coverage"])
    chk("毛利率", f["gross_margin"], lambda x: x >= g["min_gross_margin"])
    if f["roe"] is not None:
        chk("ROE", f["roe"], lambda x: x >= g["min_roe"])
    chk("营收增长", f["rev_growth"], lambda x: x >= g["min_revenue_growth"])
    if f["piotroski_n"] >= 7:
        chk("F-Score", f["piotroski"], lambda x: x >= g["min_piotroski"])
    else:
        fails.append("F-Score数据不足")
    return fails


def growth_gate(f, tech, g):
    fails = []
    def chk(name, val, ok):
        if _bad(val) or not ok(val):
            fails.append(name)
    if tech["rsi14"] > g["rsi_max"] or tech["drawdown"] < g["drawdown_min"]:
        fails.append("未达成长通道超卖标准")
    chk("市值", f["market_cap"], lambda x: x >= g["min_market_cap"])
    chk("营收增速", f["rev_growth_ttm"], lambda x: x >= g["min_revenue_growth"])
    chk("毛利率", f["gross_margin"], lambda x: x >= g["min_gross_margin"])
    chk("40法则", f["rule_of_40"], lambda x: x >= g["min_rule_of_40"])
    chk("现金跑道", f["runway_years"], lambda x: x >= g["min_runway_years"])
    chk("股权稀释", f["share_growth"], lambda x: x <= g["max_share_dilution"])
    if g["require_margin_improving"]:
        if f["op_margin"] is None or f["op_margin_prev"] is None or f["op_margin"] <= f["op_margin_prev"]:
            fails.append("利润率未改善")
    burning = not _bad(f["fcf_margin"]) and f["fcf_margin"] < 0
    if g["require_net_cash_if_burning"] and burning and f["cash"] < f["debt"]:
        fails.append("烧钱且净负债")
    return fails


# ============================== 4. 价值陷阱预警 ==============================
def red_flags(t, tech_row, sector, close, cfg):
    flags, notes = [], {}
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

    etf = SECTOR_ETF.get(sector)
    if etf and etf in close.columns:
        e = close[etf].dropna()
        rel = tech_row["ret_21d"] - (e.iloc[-1] / e.iloc[-22] - 1)
        notes["rel_sector_21d"] = rel
        if rel <= cfg["idio_drop_flag"]:
            flags.append(f"跑输{etf} {rel:.0%}")

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


# ============================== 5. 排序与报告 ==============================
def rank_pct(s, ascending=True):
    return s.rank(pct=True, ascending=ascending).fillna(0)


def score_value(df):
    q = df["piotroski"] / 9 + df["fcf_margin"].clip(0, 0.4) + df["gross_margin"].clip(0, 0.8) / 2
    return (0.35 * rank_pct(df["oversold_score"]) + 0.35 * rank_pct(q)
            + 0.30 * rank_pct(df["fcf_yield"]) - 0.10 * df["n_flags"])


def score_growth(df):
    margin_delta = (df["op_margin"] - df["op_margin_prev"]).astype(float)
    q = rank_pct(df["rule_of_40"]) + rank_pct(df["gross_margin"]) + rank_pct(margin_delta)
    cheap = df["ev_sales"] / (df["rev_growth_ttm"].clip(lower=0.01) * 100)   # 增长调整后的 EV/Sales，越低越便宜
    return (0.35 * rank_pct(df["oversold_score"]) + 0.35 * rank_pct(q)
            + 0.30 * rank_pct(cheap, ascending=False) - 0.10 * df["n_flags"])


COLS = {
    "value": ["track", "name", "sector", "price", "score", "rsi14", "drawdown", "fcf_yield", "fcf_margin",
              "net_debt_ebitda", "interest_cover", "gross_margin", "roe", "rev_growth", "piotroski", "fwd_pe",
              "eps_rev_0y", "eps_rev_+1y", "rel_sector_21d", "next_earnings", "flags", "headlines"],
    "growth": ["track", "name", "sector", "industry", "price", "score", "rsi14", "drawdown", "rev_growth_ttm",
               "gross_margin", "fcf_margin", "rule_of_40", "op_margin", "op_margin_prev", "runway_years",
               "share_growth", "ev_sales", "market_cap", "eps_rev_+1y", "rel_sector_21d", "next_earnings",
               "flags", "headlines"],
}
PCT = ["drawdown", "vs_200dma", "fcf_yield", "fcf_margin", "gross_margin", "roe", "rev_growth", "rev_growth_ttm",
       "rule_of_40", "op_margin", "op_margin_prev", "share_growth", "eps_rev_0y", "eps_rev_+1y", "rel_sector_21d"]
NUM = ["price", "score", "rsi14", "net_debt_ebitda", "interest_cover", "fwd_pe", "ev_sales", "runway_years"]


def write_html(parts, path, stamp):
    css = ("<style>body{font-family:-apple-system,'PingFang SC',Segoe UI,sans-serif;margin:24px;color:#1a1a1a}"
           "table{border-collapse:collapse;font-size:12px}th,td{padding:6px 8px;border-bottom:1px solid #eee;"
           "text-align:right;white-space:nowrap}td:last-child{white-space:normal;max-width:420px;text-align:left}"
           "th{background:#fafafa;position:sticky;top:0}h3{margin-top:32px}</style>")
    body = ""
    titles = {"value": "价值通道 · 已盈利、财务稳健", "growth": "成长通道 · 高增长、可能尚未盈利（风险更高，宜小仓位）"}
    for track, df in parts.items():
        if df is None or df.empty:
            body += f"<h3>{titles[track]}</h3><p>今天没有符合条件的股票。</p>"
            continue
        fmt = {c: "{:.1%}" for c in PCT if c in df.columns}
        fmt.update({c: "{:.2f}" for c in NUM if c in df.columns})
        if "market_cap" in df.columns:
            fmt["market_cap"] = lambda x: f"{x/1e9:.1f}B" if pd.notna(x) else "—"
        styled = (df.style.format(fmt, na_rep="—")
                  .background_gradient(subset=["score"], cmap="Greens")
                  .map(lambda v: "color:#b42318;font-weight:600" if isinstance(v, str) and v else "",
                       subset=["flags"]))
        body += f"<h3>{titles[track]} · {len(df)} 只</h3>{styled.to_html()}"
    Path(path).write_text(f"<html><head><meta charset='utf-8'>{css}</head><body>"
                          f"<h2>超卖 + 质量筛选 · {stamp}</h2>"
                          f"<p>候选清单，不构成投资建议。红色预警项需人工核实下跌原因。</p>{body}</body></html>",
                          encoding="utf-8")


# ============================== 主流程 ==============================
def run(cfg=CONFIG):
    t0 = time.time()
    print("[1] 构建股票池")
    universe = build_universe(cfg)
    print(f"    合计 {len(universe)} 只")

    close, vol = download_prices(universe + list(SECTOR_ETF.values()) + ["SPY"], cfg["price_period"])
    uni = [c for c in universe if c in close.columns]
    tech = technical_table(close[uni], vol[uni], cfg["min_dollar_volume"])
    cands = oversold_filter(tech, cfg)
    print(f"[2] 有效价格数据 {len(tech)} 只，技术面超卖 {len(cands)} 只")

    passed, rejected = [], []
    for i, tk in enumerate(cands.index, 1):
        tr = cands.loc[tk]
        try:
            f = fundamentals(tk)
        except Exception as e:
            rejected.append({"ticker": tk, "fails": f"数据错误: {e}"})
            continue
        if f["sector"] in cfg["exclude_sectors"]:
            continue
        track, reasons = None, {}
        if cfg["value"]["enabled"]:
            vf = value_gate(f, tr, cfg["value"])
            if not vf:
                track = "value"
            reasons["价值"] = vf
        if track is None and cfg["growth"]["enabled"]:
            gf = growth_gate(f, tr, cfg["growth"])
            if not gf:
                track = "growth"
            reasons["成长"] = gf
        if track is None:
            rejected.append({"ticker": tk, "name": f["name"],
                             **{f"{k}通道未通过": ", ".join(v) for k, v in reasons.items()}})
        else:
            flags, notes = red_flags(f.pop("_ticker_obj"), tr, f["sector"], close, cfg)
            passed.append({"ticker": tk, "track": track, **f, **tr.to_dict(), **notes,
                           "flags": "; ".join(flags), "n_flags": len(flags)})
        print(f"    {i}/{len(cands)} {tk}: {track or '淘汰'}")
        time.sleep(cfg["sleep_between_calls"])

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(exist_ok=True)
    stamp = dt.date.today().isoformat()
    pd.DataFrame(rejected).to_csv(out_dir / f"rejected_{stamp}.csv", index=False)

    if not passed:
        print("[3] 今天没有同时满足超卖 + 质量门槛的股票。")
        return None

    df = pd.DataFrame(passed).set_index("ticker")
    parts = {}
    for track, scorer in (("value", score_value), ("growth", score_growth)):
        d = df[df["track"] == track].copy()
        if len(d):
            d["score"] = scorer(d)
            d = d.sort_values("score", ascending=False)
            d = d[[c for c in COLS[track] if c in d.columns]]
        parts[track] = d
    print(f"[3] 价值通道 {len(parts['value'])} 只，成长通道 {len(parts['growth'])} 只，用时 {time.time() - t0:.0f}s")

    combined = pd.concat([p for p in parts.values() if len(p)], sort=False)
    combined.to_csv(out_dir / f"screen_{stamp}.csv")
    write_html(parts, out_dir / f"screen_{stamp}.html", stamp)
    for track, d in parts.items():
        if len(d):
            print(f"\n--- {track} ---")
            print(d[["name", "score", "rsi14", "drawdown", "flags"]].head(15).to_string())
    return combined


if __name__ == "__main__":
    run()
