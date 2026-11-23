#!/usr/bin/env python3
"""最佳阻尼係數隨已賽場數的變化（2021–2025 逐日滾動預測）。

每場比賽依「兩隊中較少的已賽場數」每 10 場分一組，各組分別找出
log loss 最低的單一阻尼係數 α，並與 d = (G/2) / (G/2 + k) 的理論曲線比較
（G = 已賽場數，G/2 約為輸球場數）。輸出 damping_by_games.csv。

執行： python3 damping_by_games.py   （需先有 mlb_games_YYYY.csv 快取）
"""
import csv
import os

import numpy as np

import damping_backtest as b
import damping_stage_backtest as st

BIN = 10
ALPHA = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]]), 3)
K = 5


def main():
    b.C_GRID = np.array([])
    b.ALPHA_GRID = ALPHA
    st.STAGES = [(g, g + BIN) for g in range(10, 160, BIN)]
    res = [st.run_season(s) for s in b.SEASONS]
    cnt = sum(r['cnt'] for r in res)
    ll = sum(r['ll'] for r in res)

    # 每組用自助法（依比賽日重抽太繁瑣，這裡依球季重抽）估計最佳 α 的範圍
    rng = np.random.default_rng(0)
    boot = np.zeros((500, len(st.STAGES)))
    for i in range(500):
        pick = rng.integers(0, len(res), len(res))
        c = sum(res[j]['cnt'] for j in pick)
        l = sum(res[j]['ll'] for j in pick)
        boot[i] = ALPHA[(l / np.maximum(c, 1)[:, None]).argmin(axis=1)]

    out = os.path.join(b.HERE, 'damping_by_games.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['games_from', 'games_to', 'n_games', 'best_alpha', 'alpha_p10', 'alpha_p90',
                    'best_logloss', 'formula_d_k5'])
        print(f'{"場數":>9} {"比賽數":>6} {"最佳α":>6} {"10–90%":>12} {"公式 d(k=5)":>11}')
        for s, (lo, hi) in enumerate(st.STAGES):
            if cnt[s] == 0:
                continue
            l = ll[s] / cnt[s]
            mid = (lo + hi) / 2
            fd = (mid / 2) / (mid / 2 + K)
            p10, p90 = np.percentile(boot[:, s], [10, 90])
            w.writerow([lo, hi, int(cnt[s]), ALPHA[l.argmin()], p10, p90, f'{l.min():.5f}', f'{fd:.3f}'])
            print(f'{lo:>4}–{hi:<4} {int(cnt[s]):>6} {ALPHA[l.argmin()]:>6.3f} {p10:>5.2f}–{p90:<5.2f} {fd:>11.3f}')
    print(f'已存檔：{out}')


if __name__ == '__main__':
    main()
