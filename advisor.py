#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC / ETH 日线级别趋势跟踪顾问
================================
用法:
    python advisor.py            # 每晚运行: 输出今日操作建议并写入 reports/
    python advisor.py backtest   # 2020-01-01 至今回测 (策略 vs 一直持有)
    python advisor.py both       # 两者都跑

策略 (由 research.py 对 2020~今 的 300+ 组策略做样本内/样本外/前推检验后选定, 见 reports/research_report.md):
  * 均线集成 (MA Ensemble): 7 条 SMA (50/75/100/125/150/175/200), 收盘在几条之上就持有几分之几仓位
  * 目标仓位量化到 25% 一档 (0/25/50/75/100), 变动不足一档不动 —— 控制换手
  * 没有任何"调出来"的阈值: 均线周期等距覆盖中长期, 因此对参数不敏感, 过拟合风险最低
  * 本质是把"200 日线上方持有"这个经典规则平滑化: 不再一根线决定全部仓位, 而是逐级加减
  * RSI / ATR / Mayer / 吊灯位仅作为参考信息展示, 不参与决策; 量能经检验无增量价值, 不计算
只用已收盘的日 K (UTC 日线, 北京时间每天 08:00 收盘), 晚上 21:00 跑时信号稳定不会漂移。
"""
import json
import os
import sys
import time
import datetime as dt

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
REPORT_DIR = os.path.join(BASE, "reports")
STATE_FILE = os.path.join(BASE, "state.json")
CONFIG_FILE = os.path.join(BASE, "config.json")

DEFAULT_CONFIG = {
    "coins": ["BTC", "ETH"],
    "capital_per_coin_usdt": {"BTC": 10000, "ETH": 10000},
    "backtest_start": "2020-01-01",
    "fee_rate": 0.001,
    "params": {
        "ma_periods": [50, 75, 100, 125, 150, 175, 200],
        "step_pct": 25,
        "sma_long": 200, "ema_fast": 20, "ema_mid": 50,
        "rsi_n": 14, "atr_n": 14, "donchian_n": 20,
        "chandelier_n": 22, "chandelier_mult": 3.0
    }
}

# --------------------------------------------------------------------------- #
# 配置 / 状态
# --------------------------------------------------------------------------- #
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        merged.update({k: v for k, v in cfg.items() if k != "params"})
        merged["params"].update(cfg.get("params", {}))
        return merged
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
    return json.loads(json.dumps(DEFAULT_CONFIG))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# 数据获取: OKX 优先, Coinbase 备用; 本地 CSV 缓存增量更新
# --------------------------------------------------------------------------- #
def _okx_fetch(coin, since_ms=None):
    """OKX history-candles, 向前翻页直到 since_ms"""
    inst = f"{coin}-USDT"
    rows = []
    after = None
    for _ in range(200):
        p = {"instId": inst, "bar": "1Dutc", "limit": "100"}
        if after:
            p["after"] = after
        r = requests.get("https://www.okx.com/api/v5/market/history-candles",
                         params=p, timeout=20)
        r.raise_for_status()
        d = r.json().get("data", [])
        if not d:
            break
        for c in d:
            ts, o, h, l, cl, vol = int(c[0]), c[1], c[2], c[3], c[4], c[5]
            confirmed = c[8] == "1" if len(c) > 8 else True
            if confirmed:
                rows.append((ts, float(o), float(h), float(l), float(cl), float(vol)))
        after = d[-1][0]
        if since_ms and int(d[-1][0]) <= since_ms:
            break
        time.sleep(0.12)
    if not rows:
        raise RuntimeError("OKX 无数据")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df


def _coinbase_fetch(coin, since_ms=None):
    prod = f"{coin}-USD"
    end = int(time.time())
    start_floor = int(since_ms / 1000) if since_ms else int(dt.datetime(2017, 1, 1).timestamp())
    rows = []
    while end > start_floor:
        start = max(start_floor, end - 300 * 86400)
        r = requests.get(
            f"https://api.exchange.coinbase.com/products/{prod}/candles",
            params={"granularity": 86400, "start": dt.datetime.utcfromtimestamp(start).isoformat(),
                    "end": dt.datetime.utcfromtimestamp(end).isoformat()}, timeout=20)
        r.raise_for_status()
        d = r.json()
        if not d:
            break
        for c in d:  # [time, low, high, open, close, volume]
            rows.append((int(c[0]) * 1000, float(c[3]), float(c[2]), float(c[1]), float(c[4]), float(c[5])))
        end = start - 1
        time.sleep(0.25)
    if not rows:
        raise RuntimeError("Coinbase 无数据")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    # Coinbase 最后一根可能未收盘: 去掉今天 (UTC) 的那根
    today_ms = int(dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()) * 1000
    return df[df.ts < today_ms]


def get_daily(coin):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{coin}_1d.csv")
    cached = None
    since_ms = None
    if os.path.exists(path):
        cached = pd.read_csv(path)
        if len(cached):
            since_ms = int(cached.ts.max()) - 5 * 86400_000  # 重叠几天防漏
    df = None
    for name, fn in (("OKX", _okx_fetch), ("Coinbase", _coinbase_fetch)):
        try:
            df = fn(coin, since_ms)
            src = name
            break
        except Exception as e:  # noqa
            print(f"  [{coin}] {name} 获取失败: {e}")
    if df is None:
        if cached is not None:
            print(f"  [{coin}] 全部数据源失败, 使用本地缓存")
            df, src = cached, "cache"
        else:
            raise RuntimeError(f"{coin}: 无法获取数据")
    if cached is not None and src != "cache":
        df = pd.concat([cached, df])
    df = (df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True))
    df.to_csv(path, index=False)
    df["date"] = pd.to_datetime(df.ts, unit="ms", utc=True).dt.date
    df = df.set_index(pd.to_datetime(df.ts, unit="ms", utc=True)).drop(columns="ts")
    return df, src


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def add_indicators(df, P):
    c, h, l = df.close, df.high, df.low
    df["sma200"] = c.rolling(P["sma_long"]).mean()
    df["sma200_slope"] = df.sma200 / df.sma200.shift(20) - 1
    df["ema20"] = c.ewm(span=P["ema_fast"], adjust=False).mean()
    df["ema50"] = c.ewm(span=P["ema_mid"], adjust=False).mean()
    for n in P["ma_periods"]:
        df[f"ma{n}"] = c.rolling(n).mean()
    # RSI (Wilder)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / P["rsi_n"], adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / P["rsi_n"], adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df.macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df.macd - df.macd_sig
    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / P["atr_n"], adjust=False).mean()
    df["atr_pct"] = df.atr / c
    # Donchian
    n = P["donchian_n"]
    df["dc_high"] = h.rolling(n).max().shift(1)
    df["dc_low"] = l.rolling(n).min().shift(1)
    df["dc_mid"] = (df.dc_high + df.dc_low) / 2
    # Chandelier stop
    df["chandelier"] = c.rolling(P["chandelier_n"]).max() - P["chandelier_mult"] * df.atr
    # Mayer multiple
    df["mayer"] = c / df.sma200
    # 距离 200 日线
    df["dist_sma200"] = c / df.sma200 - 1
    return df


def ma_votes(r, P):
    """返回 (在其上的均线数, 明细dict)"""
    detail = {f"SMA{n}": (float(r[f"ma{n}"]), bool(r.close > r[f"ma{n}"])) for n in P["ma_periods"]}
    return sum(1 for _, ok in detail.values() if ok), detail


def raw_target(votes, P):
    return 100.0 * votes / len(P["ma_periods"])


def next_target(prev, raw, P):
    """目标仓位量化到 step_pct 一档, 变动不足一档不动"""
    step = P["step_pct"]
    if abs(raw - prev) >= step - 1e-9:
        return int(round(raw / step) * step)
    return int(prev)


def run_signals(df, P):
    """逐日生成目标仓位序列 (用于回测和当日建议的一致性)"""
    targets = []
    prev = 0
    last_n = max(P["ma_periods"])
    for _, r in df.iterrows():
        if np.isnan(r[f"ma{last_n}"]):
            targets.append(0)
            continue
        v, _ = ma_votes(r, P)
        prev = next_target(prev, raw_target(v, P), P)
        targets.append(prev)
    df["target"] = targets
    return df


# --------------------------------------------------------------------------- #
# 回测
# --------------------------------------------------------------------------- #
def backtest(df, start, fee):
    d = df.copy()
    d["ret"] = d.close.pct_change()
    d["pos"] = d.target.shift(1).fillna(0) / 100.0  # 收盘出信号, 下一根生效
    d["turnover"] = d.pos.diff().abs().fillna(0)
    d["strat_ret"] = d.pos * d.ret - d.turnover * fee
    d = d[d.index >= pd.Timestamp(start, tz="UTC")].copy()
    d["eq_strat"] = (1 + d.strat_ret.fillna(0)).cumprod()
    d["eq_hold"] = (1 + d.ret.fillna(0)).cumprod()

    def stats(eq, rets):
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        dd = (eq / eq.cummax() - 1).min()
        sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
        return {"总收益": eq.iloc[-1] - 1, "年化": cagr, "最大回撤": dd, "夏普": sharpe}

    s1 = stats(d.eq_strat, d.strat_ret.fillna(0))
    s2 = stats(d.eq_hold, d.ret.fillna(0))
    trades = int((d.turnover > 0).sum())
    exposure = float(d.pos.mean())
    yearly = pd.DataFrame({
        "策略": d.groupby(d.index.year).strat_ret.apply(lambda x: (1 + x.fillna(0)).prod() - 1),
        "持有": d.groupby(d.index.year).ret.apply(lambda x: (1 + x.fillna(0)).prod() - 1),
    })
    return d, s1, s2, trades, exposure, yearly


def fmt_pct(x):
    return f"{x * 100:+.1f}%"


def print_backtest(coin, res):
    d, s1, s2, trades, exposure, yearly = res
    print(f"\n================ {coin} 回测 {d.index[0].date()} ~ {d.index[-1].date()} ================")
    print(f"{'':10s}{'策略':>14s}{'一直持有':>14s}")
    for k in s1:
        v1, v2 = s1[k], s2[k]
        if k == "夏普":
            print(f"{k:10s}{v1:>14.2f}{v2:>14.2f}")
        else:
            print(f"{k:10s}{fmt_pct(v1):>14s}{fmt_pct(v2):>14s}")
    print(f"调仓次数: {trades}   平均仓位: {exposure * 100:.0f}%")
    print("分年度收益:")
    for y, row in yearly.iterrows():
        print(f"  {y}: 策略 {fmt_pct(row['策略']):>8s}   持有 {fmt_pct(row['持有']):>8s}")
    os.makedirs(REPORT_DIR, exist_ok=True)
    d[["close", "target", "pos", "eq_strat", "eq_hold"]].to_csv(
        os.path.join(REPORT_DIR, f"backtest_{coin}.csv"))


# --------------------------------------------------------------------------- #
# 每日建议
# --------------------------------------------------------------------------- #
def advise(coin, df, cfg, state, src):
    P = cfg["params"]
    r = df.iloc[-1]
    votes, detail = ma_votes(r, P)
    n_ma = len(P["ma_periods"])
    raw = raw_target(votes, P)
    cap = cfg["capital_per_coin_usdt"].get(coin, 0)

    st = state.get(coin, {})
    # 用户实际仓位优先; 没填则用策略自身回放出来的上一日目标
    prev = st.get("current_pct", int(df.target.iloc[-2]) if len(df) > 1 else 0)
    target = next_target(prev, raw, P)

    if target > prev:
        action = f"买入 —— 仓位从 {prev}% 提到 {target}%"
    elif target < prev:
        action = f"卖出 —— 仓位从 {prev}% 降到 {target}%"
    else:
        action = f"持有不动 —— 维持 {target}% 仓位"
    delta_usdt = (target - prev) / 100 * cap
    qty = delta_usdt / r.close if r.close else 0

    # 离下一次变动还有多远: 最近的一条会被穿越的均线
    above = sorted([(v, k) for k, (v, ok) in detail.items() if ok], reverse=True)     # 最高的支撑
    below = sorted([(v, k) for k, (v, ok) in detail.items() if not ok])               # 最低的压力
    regime = "多头 (200日线上方)" if r.close > r.sma200 else "空头 (200日线下方)"
    lines = []
    lines.append(f"## {coin}  —  {action}")
    lines.append("")
    lines.append(f"- 数据: {src}  最新已收盘日K: {df.index[-1].date()} (UTC)  收盘价: {r.close:,.2f}")
    lines.append(f"- 大势: {regime}   Mayer倍数 {r.mayer:.2f}   RSI14 {r.rsi:.1f}   ATR14 {r.atr:,.0f} ({r.atr_pct * 100:.1f}%/日)")
    lines.append(f"- 均线投票: **{votes} / {n_ma}** 条均线之上 → 原始目标 {raw:.0f}% → 量化到 **{target}%**")
    for k, (v, ok) in detail.items():
        lines.append(f"    - {'✅' if ok else '❌'} {k} {v:,.0f} ({(r.close / v - 1) * 100:+.1f}%)")
    lines.append(f"- 目标仓位: **{target}%**  (当前记录 {prev}%)")
    if abs(delta_usdt) > 0:
        lines.append(f"- 金额: 约 {delta_usdt:+,.0f} USDT  ≈ {qty:+.5f} {coin}  (按本金 {cap:,} USDT 计)")
    if above:
        lines.append(f"- 下方最近支撑: {above[0][1]} {above[0][0]:,.0f} —— 收盘跌破则少 1 票")
    if below:
        lines.append(f"- 上方最近阻力: {below[0][1]} {below[0][0]:,.0f} —— 收盘站上则多 1 票")
    lines.append(f"- 参考: 吊灯止损位 {r.chandelier:,.0f}   20日通道 {r.dc_low:,.0f} ~ {r.dc_high:,.0f}")
    if target == 100:
        note = "全部均线之上, 趋势完整, 满仓拿住。"
    elif target >= 50:
        note = "多数均线之上, 趋势尚可但不完整; 逐级跟随, 不预测。"
    elif target > 0:
        note = "多数均线之下, 轻仓观察; 站回均线再逐级加。"
    else:
        note = "全部均线之下, 空仓等待。"
    lines.append(f"- 解读: {note}")
    lines.append("")

    state[coin] = {"current_pct": target, "last_run": df.index[-1].date().isoformat(),
                   "last_close": float(r.close), "votes": votes,
                   "note": "current_pct 为策略假定你已执行后的仓位; 若你实际未执行, 请手动改成真实仓位"}
    return "\n".join(lines)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "advise"
    cfg = load_config()
    P = cfg["params"]
    state = load_state()
    now_bj = dt.datetime.utcnow() + dt.timedelta(hours=8)
    header = [f"# 每日操作建议  北京时间 {now_bj:%Y-%m-%d %H:%M}", "",
              "> 规则: 7 条均线 (50~200 日) 投票决定仓位, 25% 一档. 信号基于已收盘的 UTC 日K, 一天只变一次.", ""]
    body = []
    for coin in cfg["coins"]:
        print(f"获取 {coin} 日线...")
        df, src = get_daily(coin)
        df = add_indicators(df, P)
        df = run_signals(df, P)
        if mode in ("backtest", "both"):
            print_backtest(coin, backtest(df, cfg["backtest_start"], cfg["fee_rate"]))
        if mode in ("advise", "both"):
            body.append(advise(coin, df, cfg, state, src))
    if body:
        report = "\n".join(header + body)
        print("\n" + report)
        os.makedirs(REPORT_DIR, exist_ok=True)
        fn = os.path.join(REPORT_DIR, f"advice_{now_bj:%Y-%m-%d}.md")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(report)
        with open(os.path.join(REPORT_DIR, "latest.md"), "w", encoding="utf-8") as f:
            f.write(report)
        save_state(state)
        print(f"\n报告已保存: {fn}")


if __name__ == "__main__":
    main()
