#!/usr/bin/env python3
"""比較各種「調整阻尼係數」的方法，找出 2021–2026 最適合的設定。

方法（前三種來自本研究，後三種參考網路文獻）：
  alpha     單一阻尼係數 α（Brin & Page 的經典 PageRank，常用 0.85）
  matrix_c  原本的阻尼矩陣 d_i = L_i / (L_i + c·mean(L))
  adapt_k   自適應阻尼 d_i = L_i / (L_i + k)，k 為固定場數
  adapt_2p  兩參數自適應 d_i = L_i^a / (L_i^a + k^a)：k = 阻尼到 0.5 的輸球數，a = 上升陡峭度
            （a 大時接近「到某個場數就跳一階」，對應分段階梯式調整阻尼的做法）
  pers_win  個人化（topic-specific）PageRank：跳轉不再均勻，而是依勝率（加 1 平滑）分配
  walker    Callaghan–Mucha–Porter 隨機漫步者：每步隨機挑目前球隊的一場比賽，
            以機率 p 改支持贏家、1−p 改支持輸家（p 取代阻尼的角色）

所有方法共用：每勝 1、主客場反機率加權、逐日滾動預測（只用當天以前的比賽）、
單場勝率 = PR(A) / (PR(A) + PR(B))、兩隊都打滿 10 場才計分。
選參數用「留一球季交叉驗證」：用其他 5 季挑參數，在剩下那一季打分數。

執行： python3 damping_methods_backtest.py [起始年 結束年]   （預設 2021 2026；需先有對應的 mlb_games_YYYY.csv 快取）
"""
import csv
import itertools
import os
import sys

import numpy as np

import damping_backtest as b

# 可用參數指定球季範圍，例如 python3 damping_methods_backtest.py 2016 2026
FIRST, LAST = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (2021, 2026)
SEASONS = list(range(FIRST, LAST + 1))
MIN_GAMES = 10

ALPHA = np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.99]]), 3)
C = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0])
K1 = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30])
K2 = list(itertools.product([5, 8, 10, 12, 15, 18, 20, 25], [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]))
PERS = ALPHA
WALK = np.round(np.arange(0.5, 1.001, 0.025), 3)

METHODS = [('alpha', [f'{a:g}' for a in ALPHA]),
           ('matrix_c', [f'{c:g}' for c in C]),
           ('adapt_k', [f'{k:g}' for k in K1]),
           ('adapt_2p', [f'k={k:g},a={a:g}' for k, a in K2]),
           ('pers_win', [f'{a:g}' for a in PERS]),
           ('walker', [f'{p:g}' for p in WALK])]


def stationary(G):
    """G：(K, n, n) 列隨機矩陣 → (K, n) 穩態分布。"""
    K, n, _ = G.shape
    M = np.transpose(G, (0, 2, 1)) - np.eye(n)[None]
    M[:, -1, :] = 1.0
    rhs = np.zeros((K, n, 1))
    rhs[:, -1, 0] = 1.0
    return np.linalg.solve(M, rhs)[:, :, 0]


def google(P, d, v):
    """d：(K, n) 每隊阻尼；v：(K, n) 跳轉分布 → G = D·P + (I−D)·1·vᵀ。"""
    return d[:, :, None] * P[None] + (1 - d[:, :, None]) * v[:, None, :]


def all_ratings(AH, AA, W, L):
    n = len(W)
    A = b.adjacency(AH, AA)                     # A[輸家, 贏家]
    rs = A.sum(axis=1)
    P = np.full((n, n), 1.0 / n)
    ok = rs > 0
    P[ok] = A[ok] / rs[ok, None]
    Lf = L.astype(float)
    uni = lambda K: np.full((K, n), 1.0 / n)
    out = []

    d = np.repeat(ALPHA[:, None], n, axis=1)
    out.append(stationary(google(P, d, uni(len(ALPHA)))))

    d = Lf[None] / (Lf[None] + C[:, None] * max(Lf.mean(), 1e-9))
    out.append(stationary(google(P, d, uni(len(C)))))

    d = Lf[None] / (Lf[None] + K1[:, None])
    out.append(stationary(google(P, d, uni(len(K1)))))

    kk = np.array([k for k, _ in K2])[:, None]
    aa = np.array([a for _, a in K2])[:, None]
    d = Lf[None] ** aa / (Lf[None] ** aa + kk ** aa)
    out.append(stationary(google(P, d, uni(len(K2)))))

    v = (W + 1) / (W + L + 2)
    v = np.repeat((v / v.sum())[None], len(PERS), axis=0)
    d = np.repeat(PERS[:, None], n, axis=1)
    out.append(stationary(google(P, d, v)))

    # 隨機漫步者：A[i, j] = i 輸給 j 的權重；i 的比賽 = 輸的（往贏家 j）+ 贏的（往輸家 j）
    Wm = A.T                                    # Wm[i, j] = i 贏 j 的權重
    g = (A.sum(axis=1) + Wm.sum(axis=1))[:, None]
    g[g == 0] = 1
    lose_to, beat = A / g, Wm / g               # 依比賽數正規化
    p = WALK[:, None, None]
    T = p * lose_to[None] + (1 - p) * beat[None]
    stay = p[:, :, 0] * beat.sum(axis=1)[None] + (1 - p[:, :, 0]) * lose_to.sum(axis=1)[None]
    T = T + stay[:, :, None] * np.eye(n)[None]
    empty = (A.sum(axis=1) + Wm.sum(axis=1)) == 0
    T[:, empty, :] = 1.0 / n
    # 極小的均勻跳轉：網絡不連通時（例如 2020 年只打同區）仍有唯一穩態，對其他年份幾乎沒有影響
    T = (1 - 1e-4) * T + 1e-4 / n
    out.append(stationary(T))
    return np.vstack(out)


def run_season(season):
    games = b.load_csv(os.path.join(b.HERE, f'mlb_games_{season}.csv'))
    ids = sorted({int(g['away_id']) for g in games} | {int(g['home_id']) for g in games})
    ix = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    K = sum(len(p) for _, p in METHODS)
    AH, AA = np.zeros((n, n)), np.zeros((n, n))
    W, L = np.zeros(n), np.zeros(n)
    N, hits, ll = 0, np.zeros(K), np.zeros(K)
    pct_hits = pct_ll = 0.0
    by_date = {}
    for g in games:
        by_date.setdefault(g['date'], []).append(g)
    for date in sorted(by_date):
        today = [(ix[int(g['away_id'])], ix[int(g['home_id'])],
                  int(g['away_score']) > int(g['home_score'])) for g in by_date[date]]
        played = W + L
        elig = [t for t in today if min(played[t[0]], played[t[1]]) >= MIN_GAMES]
        if elig:
            R = all_ratings(AH, AA, W, L)
            for a, h, aw in elig:
                p = R[:, a] / (R[:, a] + R[:, h])
                N += 1
                hits += np.where(p == 0.5, 0.5, (p > 0.5) == aw)
                ll += -np.log(np.clip(np.where(aw, p, 1 - p), 1e-12, 1))
                x, z = W[a] / played[a], W[h] / played[h]
                q = x * (1 - z) / (x * (1 - z) + z * (1 - x)) if 0 < x < 1 and 0 < z < 1 else 0.5
                pct_hits += 0.5 if q == 0.5 else float((q > 0.5) == aw)
                q = min(max(q, 1e-6), 1 - 1e-6)
                pct_ll += -np.log(q if aw else 1 - q)
        for a, h, aw in today:
            if aw:
                AA[h, a] += 1; W[a] += 1; L[h] += 1
            else:
                AH[a, h] += 1; W[h] += 1; L[a] += 1
    print(f'  {season}: {N} 場')
    return dict(season=season, n=N, hits=hits, ll=ll, pct_hits=pct_hits, pct_ll=pct_ll)


def main():
    print('逐日滾動回測中…')
    res = [run_season(s) for s in SEASONS]
    N = sum(r['n'] for r in res)
    hits = sum(r['hits'] for r in res)
    ll = sum(r['ll'] for r in res)

    rows = []
    start = 0
    print(f'\n{SEASONS[0]}–{SEASONS[-1]} 共 {N} 場（兩隊皆 ≥ {MIN_GAMES} 場）')
    print(f'對照：勝率較高者勝（log5） 命中率 {sum(r["pct_hits"] for r in res) / N:.4f}，'
          f'log loss {sum(r["pct_ll"] for r in res) / N:.5f}；丟銅板 log loss {np.log(2):.5f}\n')
    print(f'{"方法":<10} {f"{len(SEASONS)}季最佳參數":<14} {"log loss":>9} {"命中率":>7}   {"樣本外 log loss":>14} {"樣本外命中率":>11}   各季挑出的參數')
    for name, params in METHODS:
        sl = slice(start, start + len(params))
        start += len(params)
        l_all, h_all = ll[sl] / N, hits[sl] / N
        best = l_all.argmin()
        cv_l = cv_h = 0.0
        picks = []
        for r in res:
            j = (ll[sl] - r['ll'][sl]).argmin()
            picks.append(params[j])
            cv_l += r['ll'][sl][j]
            cv_h += r['hits'][sl][j]
        print(f'{name:<10} {params[best]:<14} {l_all[best]:>9.5f} {h_all[best]:>7.4f}   '
              f'{cv_l / N:>14.5f} {cv_h / N:>11.4f}   {" / ".join(picks)}')
        rows.append([name, params[best], f'{l_all[best]:.6f}', f'{h_all[best]:.5f}',
                     f'{cv_l / N:.6f}', f'{cv_h / N:.5f}', ' / '.join(picks)]
                    + [f'{r["ll"][sl][best] / r["n"]:.6f}' for r in res])
    name = 'damping_methods_results.csv' if (FIRST, LAST) == (2021, 2026) else f'damping_methods_results_{FIRST}_{LAST}.csv'
    out = os.path.join(b.HERE, name)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'best_param_2021_2026', 'logloss', 'accuracy', 'cv_logloss', 'cv_accuracy',
                    'cv_picks'] + [f'logloss_{s}_at_best' for s in SEASONS])
        w.writerows(rows)
    print(f'\n已存檔：{out}')


if __name__ == '__main__':
    main()
