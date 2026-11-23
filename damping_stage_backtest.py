#!/usr/bin/env python3
"""分段阻尼回測：開季、季中、季末是否該用不同的阻尼值？

沿用 damping_backtest.py 的模型與逐日滾動（walk-forward）預測，
但把每一場依「兩隊中較少的已賽場數」分成幾個階段，各階段分開計分，
再比較：
    單一設定：整季用同一個 c（或 α）
    分段設定：每個階段各自挑最好的 c（或 α）
兩者都用「留一球季交叉驗證」（用其他 4 季挑參數、在剩下那季驗證），
避免分段版因為參數比較多而在樣本內看起來比較好。

執行： python3 damping_stage_backtest.py   （需先有 mlb_games_YYYY.csv 快取）
"""
import os

import numpy as np

import damping_backtest as b

STAGES = [(10, 40), (40, 80), (80, 999)]      # 依兩隊較少的已賽場數分段
# 季中、季末的最佳值落在原本網格的邊界（c 最小、α 最大），所以往外延伸
b.C_GRID = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
b.ALPHA_GRID = np.array([0.3, 0.45, 0.6, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999])
MIN_GAMES = STAGES[0][0]


def run_season(season):
    games = b.load_csv(os.path.join(b.HERE, f'mlb_games_{season}.csv'))
    ids = sorted({int(g['away_id']) for g in games} | {int(g['home_id']) for g in games})
    ix = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    K = len(b.C_GRID) + len(b.ALPHA_GRID)
    S = len(STAGES)
    AH, AA = np.zeros((n, n)), np.zeros((n, n))
    W, L = np.zeros(n), np.zeros(n)
    cnt = np.zeros(S)
    hits, ll = np.zeros((S, K)), np.zeros((S, K))
    pct_hits = np.zeros(S)

    by_date = {}
    for g in games:
        by_date.setdefault(g['date'], []).append(g)
    for date in sorted(by_date):
        today = [(ix[int(g['away_id'])], ix[int(g['home_id'])],
                  int(g['away_score']) > int(g['home_score'])) for g in by_date[date]]
        played = W + L
        pr = None
        for a, h, aw in today:
            m = min(played[a], played[h])
            s = next((i for i, (lo, hi) in enumerate(STAGES) if lo <= m < hi), None)
            if s is None:
                continue
            if pr is None:
                pr = b.pagerank_batch(b.adjacency(AH, AA), b.damping_sets(L))
            p = pr[:, a] / (pr[:, a] + pr[:, h])
            cnt[s] += 1
            hits[s] += np.where(p == 0.5, 0.5, (p > 0.5) == aw)
            ll[s] += -np.log(np.where(aw, p, 1 - p))
            pa, ph = W[a] / played[a], W[h] / played[h]
            pct_hits[s] += 0.5 if pa == ph else float((pa > ph) == aw)
        for a, h, aw in today:
            if aw:
                AA[h, a] += 1; W[a] += 1; L[h] += 1
            else:
                AH[a, h] += 1; W[h] += 1; L[a] += 1
    return dict(season=season, cnt=cnt, hits=hits, ll=ll, pct_hits=pct_hits)


def main():
    res = [run_season(s) for s in b.SEASONS]
    nC = len(b.C_GRID)
    cnt = sum(r['cnt'] for r in res)
    hits = sum(r['hits'] for r in res)
    ll = sum(r['ll'] for r in res)
    pct = sum(r['pct_hits'] for r in res)
    N = cnt.sum()

    for name, sl, grid in (('阻尼矩陣 c', slice(0, nC), b.C_GRID),
                           ('單一阻尼係數 α', slice(nC, None), b.ALPHA_GRID)):
        print(f'\n=== {name}：各階段的最佳值（{b.SEASONS[0]}–{b.SEASONS[-1]}）===')
        print(f'{"階段（較少場數）":<14} {"場數":>6} {"最佳(logloss)":>13} {"log loss":>9} '
              f'{"最佳(命中)":>10} {"命中率":>7} {"勝率較高者":>9}')
        for s, (lo, hi) in enumerate(STAGES):
            l, a = ll[s, sl] / cnt[s], hits[s, sl] / cnt[s]
            print(f'{f"{lo}–{hi if hi < 999 else 162} 場":<16} {int(cnt[s]):>6} {grid[l.argmin()]:>13.3f} '
                  f'{l.min():>9.5f} {grid[a.argmax()]:>10.3f} {a.max():>7.4f} {pct[s] / cnt[s]:>9.4f}')
        print('各球季、各階段 log loss 最佳值（看穩不穩定）：')
        for r in res:
            best = [grid[(r['ll'][s, sl] / r['cnt'][s]).argmin()] for s in range(len(STAGES))]
            print(f'    {r["season"]}：' + '  '.join(f'{v:.3f}' for v in best))

        # 留一球季交叉驗證：單一設定 vs 分段設定
        cv = {'single_ll': 0.0, 'stage_ll': 0.0, 'single_hit': 0.0, 'stage_hit': 0.0}
        for r in res:
            oth_ll = ll[:, sl] - r['ll'][:, sl]                  # (S, K')
            oth_hit = hits[:, sl] - r['hits'][:, sl]
            k1 = oth_ll.sum(axis=0).argmin()
            cv['single_ll'] += r['ll'][:, sl][:, k1].sum()
            cv['stage_ll'] += sum(r['ll'][s, sl][oth_ll[s].argmin()] for s in range(len(STAGES)))
            k2 = oth_hit.sum(axis=0).argmax()
            cv['single_hit'] += r['hits'][:, sl][:, k2].sum()
            cv['stage_hit'] += sum(r['hits'][s, sl][oth_hit[s].argmax()] for s in range(len(STAGES)))
        print(f'留一球季交叉驗證（樣本外，{int(N)} 場）：')
        print(f'    log loss：單一 {cv["single_ll"] / N:.5f}  vs  分段 {cv["stage_ll"] / N:.5f}')
        print(f'    命中率  ：單一 {cv["single_hit"] / N:.4f}  vs  分段 {cv["stage_hit"] / N:.4f}')


if __name__ == '__main__':
    main()
