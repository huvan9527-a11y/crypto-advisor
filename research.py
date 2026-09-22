#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略研究: 系统地从历史中"学习", 而不是拍脑袋
==========================================
步骤:
  1. 候选策略族 (6 族, 共 ~1300 组参数), 全部在 2020-01-01 起的数据上评估
  2. 样本内 (IS: 2020-01 ~ 2023-12) 选参, 样本外 (OOS: 2024-01 ~ 今) 验证 —— 防止"看答案做题"
  3. 滚动前推 (Walk-Forward): 每年只用之前的数据选参, 应用到下一年, 拼成一条真实可得的曲线
  4. 参数邻域稳健性: 最优点周围的参数是否也好? 孤峰 = 过拟合
  5. 输出 research_report.md
评价指标以 Calmar (年化/最大回撤) 为主 —— 因为用户是长期持有者, 最在乎"拿不拿得住"
"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advisor as A

BASE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.001
IS_START, IS_END = "2020-01-01", "2023-12-31"
OOS_START = "2024-01-01"

# --------------------------------------------------------------------------- #
# 通用: 给定 0/1 (或 0~1) 仓位序列 -> 绩效
# --------------------------------------------------------------------------- #
def perf(close, pos, start, end=None):
    ret = close.pct_change()
    pos = pos.shift(1).fillna(0)                       # 收盘出信号, 次日生效
    turn = pos.diff().abs().fillna(0)
    sr = pos * ret - turn * FEE
    m = close.index >= pd.Timestamp(start, tz="UTC")
    if end: m &= close.index <= pd.Timestamp(end, tz="UTC")
    sr = sr[m].fillna(0)
    if len(sr) < 30: return None
    eq = (1 + sr).cumprod()
    yrs = len(sr) / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = sr.mean() / sr.std() * np.sqrt(365) if sr.std() > 0 else 0
    return dict(cagr=cagr, mdd=dd, sharpe=sharpe, calmar=cagr / abs(dd) if dd < 0 else np.nan,
                trades=int((turn[m] > 0).sum()), total=eq.iloc[-1] - 1)

# --------------------------------------------------------------------------- #
# 策略族: 每个函数返回 dict{参数名: 仓位序列}
# --------------------------------------------------------------------------- #
def fam_sma(df):
    out = {}
    for n in range(50, 301, 10):
        out[f"SMA{n}"] = (df.close > df.close.rolling(n).mean()).astype(float)
    return out

def fam_sma_band(df):
    """SMA + 缓冲带: 上穿 (1+b) 买, 下穿 (1-b) 卖. 减少来回打脸"""
    out = {}
    for n in [100, 150, 200, 250]:
        sma = df.close.rolling(n).mean()
        for b in [0.02, 0.05, 0.08, 0.12]:
            up, dn = df.close > sma * (1 + b), df.close < sma * (1 - b)
            sig = pd.Series(np.where(up, 1.0, np.where(dn, 0.0, np.nan)), index=df.index).ffill().fillna(0)
            out[f"SMA{n}±{int(b*100)}%"] = sig
    return out

def fam_ema_cross(df):
    out = {}
    for f, s in itertools.product([10, 20, 30, 50], [50, 100, 150, 200]):
        if f >= s: continue
        ef, es = df.close.ewm(span=f, adjust=False).mean(), df.close.ewm(span=s, adjust=False).mean()
        out[f"EMA{f}/{s}"] = (ef > es).astype(float)
    return out

def fam_donchian(df):
    out = {}
    for n_in, n_out in itertools.product([20, 40, 55, 80, 100], [10, 20, 40, 55]):
        hi, lo = df.high.rolling(n_in).max().shift(1), df.low.rolling(n_out).min().shift(1)
        sig = pd.Series(np.where(df.close > hi, 1.0, np.where(df.close < lo, 0.0, np.nan)), index=df.index).ffill().fillna(0)
        out[f"DC{n_in}/{n_out}"] = sig
    return out

def fam_roc(df):
    out = {}
    for n in [30, 60, 90, 120, 180, 250]:
        out[f"ROC{n}"] = (df.close > df.close.shift(n)).astype(float)
    return out

def fam_score(df):
    """当前 advisor 的打分状态机, 扫阈值和止损倍数"""
    out = {}
    base = dict(A.DEFAULT_CONFIG["params"])
    grid = itertools.product([4, 5, 6], [6, 7, 8], [3, 4, 5], [1, 2, 3], [2.5, 3.0, 3.5, 4.0, 99])
    for eh, ef, xh, xa, cm in grid:
        if not (xa < xh < eh <= ef): continue
        P = dict(base); P.update(enter_half=eh, enter_full=ef, exit_half=xh, exit_all=xa, chandelier_mult=cm)
        d = A.add_indicators(df.copy(), P)
        d = A.run_signals(d, P)
        cm_s = "无止损" if cm == 99 else f"止损{cm}"
        out[f"评分 进{eh}/{ef} 出{xh}/{xa} {cm_s}"] = d.target / 100.0
    return out

FAMILIES = {"SMA过滤": fam_sma, "SMA缓冲带": fam_sma_band, "EMA金叉": fam_ema_cross,
            "唐奇安突破": fam_donchian, "动量ROC": fam_roc, "多指标评分": fam_score}

# --------------------------------------------------------------------------- #
def evaluate_all(df):
    rows = []
    for fam, fn in FAMILIES.items():
        for name, pos in fn(df).items():
            full = perf(df.close, pos, IS_START); is_ = perf(df.close, pos, IS_START, IS_END); oos = perf(df.close, pos, OOS_START)
            if not (full and is_ and oos): continue
            rows.append(dict(family=fam, name=name,
                             full_cagr=full["cagr"], full_mdd=full["mdd"], full_calmar=full["calmar"], full_sharpe=full["sharpe"], trades=full["trades"],
                             is_cagr=is_["cagr"], is_mdd=is_["mdd"], is_calmar=is_["calmar"],
                             oos_cagr=oos["cagr"], oos_mdd=oos["mdd"], oos_calmar=oos["calmar"]))
    hold = pd.Series(1.0, index=df.index)
    full, is_, oos = perf(df.close, hold, IS_START), perf(df.close, hold, IS_START, IS_END), perf(df.close, hold, OOS_START)
    rows.append(dict(family="基准", name="一直持有", full_cagr=full["cagr"], full_mdd=full["mdd"], full_calmar=full["calmar"], full_sharpe=full["sharpe"], trades=0,
                     is_cagr=is_["cagr"], is_mdd=is_["mdd"], is_calmar=is_["calmar"], oos_cagr=oos["cagr"], oos_mdd=oos["mdd"], oos_calmar=oos["calmar"]))
    return pd.DataFrame(rows)

def walk_forward(df, all_pos, metric="calmar"):
    """每年: 用 2019~上一年 (至少含2020) 的数据按 metric 选最优, 应用到当年"""
    years = list(range(2021, df.index[-1].year + 1))
    picks, eq_parts = [], []
    for y in years:
        tr_end = f"{y-1}-12-31"
        best, best_v = None, -np.inf
        for name, pos in all_pos.items():
            p = perf(df.close, pos, IS_START, tr_end)
            if p and p[metric] == p[metric] and p[metric] > best_v:
                best, best_v = name, p[metric]
        pos = all_pos[best]
        ret = df.close.pct_change(); ps = pos.shift(1).fillna(0); turn = ps.diff().abs().fillna(0)
        sr = (ps * ret - turn * FEE)
        m = (df.index >= pd.Timestamp(f"{y}-01-01", tz="UTC")) & (df.index <= pd.Timestamp(f"{y}-12-31", tz="UTC"))
        eq_parts.append(sr[m].fillna(0))
        picks.append((y, best, round(best_v, 2)))
    sr = pd.concat(eq_parts)
    eq = (1 + sr).cumprod(); yrs = len(sr) / 365.25
    return picks, dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min(),
                       sharpe=sr.mean() / sr.std() * np.sqrt(365))

def pct(x): return f"{x*100:+.0f}%"

def main():
    md = ["# 策略研究报告 —— 从 2020 年至今的历史中学习", "",
          f"评估区间: {IS_START} ~ 今。样本内 (IS) = 2020~2023, 样本外 (OOS) = 2024~今。手续费 {FEE*100:.1f}%/次。",
          "主要指标 Calmar = 年化收益 / 最大回撤 (越高越'拿得住')。", ""]
    summary = {}
    for coin in ["BTC", "ETH"]:
        print(f"==== {coin} ====")
        df, _ = A.get_daily(coin)
        res = evaluate_all(df)
        res.to_csv(os.path.join(BASE, "reports", f"research_{coin}.csv"), index=False)
        n = len(res) - 1
        hold = res[res.family == "基准"].iloc[0]
        md += [f"## {coin}  (共测试 {n} 组策略参数)", ""]

        # 1. 各族最优 (按样本内 Calmar 选, 看样本外表现)
        md += ["### 1) 每个策略族: 按样本内 Calmar 选最优 → 看样本外是否仍然有效", "",
               "| 策略族 | 样本内最优参数 | IS 年化 | IS 回撤 | IS Calmar | **OOS 年化** | **OOS 回撤** | **OOS Calmar** | 全程年化 | 全程回撤 | 调仓 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        md.append(f"| 基准 | 一直持有 | {pct(hold.is_cagr)} | {pct(hold.is_mdd)} | {hold.is_calmar:.2f} | {pct(hold.oos_cagr)} | {pct(hold.oos_mdd)} | {hold.oos_calmar:.2f} | {pct(hold.full_cagr)} | {pct(hold.full_mdd)} | 0 |")
        fam_best = {}
        for fam in FAMILIES:
            sub = res[res.family == fam].sort_values("is_calmar", ascending=False)
            b = sub.iloc[0]; fam_best[fam] = b
            md.append(f"| {fam} | {b['name']} | {pct(b.is_cagr)} | {pct(b.is_mdd)} | {b.is_calmar:.2f} | {pct(b.oos_cagr)} | {pct(b.oos_mdd)} | {b.oos_calmar:.2f} | {pct(b.full_cagr)} | {pct(b.full_mdd)} | {b.trades} |")
        md.append("")

        # 2. 族内平均 (稳健性: 整族平均好, 说明不是靠挑参数)
        md += ["### 2) 策略族整体稳健性 (族内所有参数的中位数 —— 不挑参数时的'普通水平')", "",
               "| 策略族 | 参数组数 | IS Calmar 中位 | OOS Calmar 中位 | 全程年化 中位 | 全程回撤 中位 | OOS 跑赢持有(Calmar) 比例 |", "|---|---|---|---|---|---|---|"]
        for fam in FAMILIES:
            sub = res[res.family == fam]
            beat = (sub.oos_calmar > hold.oos_calmar).mean()
            md.append(f"| {fam} | {len(sub)} | {sub.is_calmar.median():.2f} | {sub.oos_calmar.median():.2f} | {pct(sub.full_cagr.median())} | {pct(sub.full_mdd.median())} | {beat*100:.0f}% |")
        md.append("")

        # 3. SMA 邻域 (最直观的稳健性图)
        sub = res[res.family == "SMA过滤"].copy(); sub["n"] = sub.name.str[3:].astype(int)
        md += ["### 3) 参数邻域: 单条均线过滤, 周期从 50 到 300 (看是否平滑, 孤峰=过拟合)", "",
               "| 周期 | " + " | ".join(str(x) for x in sub.n[::3]) + " |", "|---|" + "---|" * len(sub.n[::3]),
               "| 全程年化 | " + " | ".join(pct(x) for x in sub.full_cagr[::3]) + " |",
               "| 全程回撤 | " + " | ".join(pct(x) for x in sub.full_mdd[::3]) + " |",
               "| OOS Calmar | " + " | ".join(f"{x:.2f}" for x in sub.oos_calmar[::3]) + " |", ""]

        # 4. 滚动前推
        all_pos = {}
        for fam, fn in FAMILIES.items(): all_pos.update(fn(df))
        picks, wf = walk_forward(df, all_pos)
        md += ["### 4) 滚动前推 (Walk-Forward): 每年只用'当时已知'的历史选最优策略, 用于下一年", "",
               "| 年份 | 当时选中的策略 | 选中时的 Calmar |", "|---|---|---|"]
        for y, nm, v in picks: md.append(f"| {y} | {nm} | {v} |")
        hold_wf = perf(df.close, pd.Series(1.0, index=df.index), "2021-01-01")
        md += ["", f"前推拼接曲线 2021~今: 年化 {pct(wf['cagr'])}, 最大回撤 {pct(wf['mdd'])}, 夏普 {wf['sharpe']:.2f}",
               f"同期一直持有: 年化 {pct(hold_wf['cagr'])}, 最大回撤 {pct(hold_wf['mdd'])}, 夏普 {hold_wf['sharpe']:.2f}", ""]

        # 5. 全程 Top 10 (仅供参考, 有后视偏差)
        top = res[res.family != "基准"].sort_values("full_calmar", ascending=False).head(10)
        md += ["### 5) 全程 Calmar 前 10 (⚠️ 含后视偏差, 仅供参考)", "", "| 策略 | 全程年化 | 全程回撤 | Calmar | 夏普 | OOS Calmar | 调仓 |", "|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            md.append(f"| {r['name']} | {pct(r.full_cagr)} | {pct(r.full_mdd)} | {r.full_calmar:.2f} | {r.full_sharpe:.2f} | {r.oos_calmar:.2f} | {r.trades} |")
        md.append("")
        summary[coin] = dict(res=res, fam_best=fam_best, hold=hold, wf=wf, picks=picks)
        for line in md[-60:]: pass
    out = os.path.join(BASE, "reports", "research_report.md")
    with open(out, "w", encoding="utf-8") as f: f.write("\n".join(md))
    print("\n".join(md))
    print(f"\n报告: {out}")

if __name__ == "__main__":
    main()
