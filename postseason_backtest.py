#!/usr/bin/env python3
"""季後賽回測（2021–2026）：用整季例行賽算出的強度，預測季後賽每一場與每個系列賽。

比較的模型（單場勝率 = PR(A) / (PR(A) + PR(B))，勝率模型用 log5）：
  two_param  兩參數自適應阻尼 d_i = L_i^3 / (L_i^3 + 12^3)（回測推薦）
  matrix_c1  原阻尼矩陣 c = 1（目前主程式）
  alpha_075  單一阻尼 α = 0.75
  winpct     例行賽勝率（log5）
  home       主隊必勝（只用於單場）

執行： python3 postseason_backtest.py   （需先有 mlb_games_YYYY.csv 與 mlb_postseason_YYYY.csv 快取）
"""
import os

import numpy as np

import damping_backtest as b

SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
ROUND = {'F': '外卡', 'D': '分區', 'L': '聯盟冠軍', 'W': '世界大賽'}


def ratings(games, ix):
    n = len(ix)
    AH, AA = np.zeros((n, n)), np.zeros((n, n))
    W, L = np.zeros(n), np.zeros(n)
    for g in games:
        a, h = ix[int(g['away_id'])], ix[int(g['home_id'])]
        if int(g['away_score']) > int(g['home_score']):
            AA[h, a] += 1; W[a] += 1; L[h] += 1
        else:
            AH[a, h] += 1; W[h] += 1; L[a] += 1
    A = b.adjacency(AH, AA)
    P = A / A.sum(axis=1, keepdims=True)
    d = np.vstack([L ** 3 / (L ** 3 + 12 ** 3),
                   L / (L + L.mean()),
                   np.full(n, 0.75)])
    G = d[:, :, None] * P[None] + (1 - d[:, :, None]) / n
    M = np.transpose(G, (0, 2, 1)) - np.eye(n)[None]
    M[:, -1, :] = 1.0
    rhs = np.zeros((3, n, 1)); rhs[:, -1, 0] = 1.0
    pr = np.linalg.solve(M, rhs)[:, :, 0]
    return pr, W / (W + L), d[0]


def game_prob(model, pr, pct, a, h):
    """回傳客隊 a 贏主隊 h 的機率。"""
    if model < 3:
        return pr[model, a] / (pr[model, a] + pr[model, h])
    x, z = pct[a], pct[h]
    return x * (1 - z) / (x * (1 - z) + z * (1 - x))


def main():
    names = ['兩參數阻尼', '原阻尼矩陣 c=1', '單一 α=0.75', '例行賽勝率']
    g_hit = np.zeros(5); g_ll = np.zeros(4); g_n = 0
    s_hit = np.zeros(4); s_n = 0
    per = []
    for season in SEASONS:
        reg = b.load_csv(os.path.join(b.HERE, f'mlb_games_{season}.csv'))
        post = b.load_csv(os.path.join(b.HERE, f'mlb_postseason_{season}.csv'))
        ids = sorted({int(g['away_id']) for g in reg} | {int(g['home_id']) for g in reg})
        ix = {t: i for i, t in enumerate(ids)}
        pr, pct, d = ratings(reg, ix)
        series = {}
        sh = np.zeros(4); sn = 0
        for g in post:
            a, h = ix[int(g['away_id'])], ix[int(g['home_id'])]
            aw = int(g['away_score']) > int(g['home_score'])
            g_n += 1
            g_hit[4] += not aw
            for m in range(4):
                p = game_prob(m, pr, pct, a, h)
                g_hit[m] += 0.5 if p == 0.5 else float((p > 0.5) == aw)
                g_ll[m] += -np.log(p if aw else 1 - p)
            key = (g['game_type'], min(a, h), max(a, h))
            s = series.setdefault(key, {'need': 0, 'w': {a: 0, h: 0}})
            s['need'] = max(s['need'], (int(g['games_in_series']) + 1) // 2)
            s['w'][a if aw else h] += 1
        for (_, x, y), s in series.items():
            win = [t for t, w in s['w'].items() if w == s['need']]
            if not win:
                continue
            lose = y if win[0] == x else x
            sn += 1
            for m in range(4):
                p = game_prob(m, pr, pct, win[0], lose)
                sh[m] += 0.5 if p == 0.5 else float(p > 0.5)
        s_hit += sh; s_n += sn
        per.append((season, sn, sh, d.min(), d.max()))

    print(f'季後賽單場（{g_n} 場）')
    print(f'  {"模型":<14} {"命中率":>7} {"log loss":>9}')
    for m in range(4):
        print(f'  {names[m]:<14} {g_hit[m] / g_n:>7.1%} {g_ll[m] / g_n:>9.5f}')
    print(f'  {"主隊必勝":<14} {g_hit[4] / g_n:>7.1%}')
    print(f'\n季後賽系列賽晉級隊（{s_n} 個）')
    for m in range(4):
        print(f'  {names[m]:<14} {int(s_hit[m])} / {s_n} = {s_hit[m] / s_n:.1%}')
    print('\n各年系列賽命中（兩參數 / 原 c=1 / α=0.75 / 勝率），以及兩參數阻尼的 d 範圍')
    for season, sn, sh, dmin, dmax in per:
        print(f'  {season}：{int(sh[0])}/{int(sh[1])}/{int(sh[2])}/{int(sh[3])} 共 {sn} 個   d = {dmin:.3f}–{dmax:.3f}')


if __name__ == '__main__':
    main()
