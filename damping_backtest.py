#!/usr/bin/env python3
"""阻尼係數回測：用前 5 個球季（2021–2025）逐日滾動預測每一場例行賽，
找出讓 PageRank 預測最準的阻尼設定。

模型與 mlb_pagerank.m 完全相同（邊：輸家 → 贏家、每勝 1、主客場反機率加權、
G = D·P + (I-D)·E），只是把阻尼設定當成要調的參數：
    matrix：阻尼矩陣  d_i = L_i / (L_i + c·mean(L))   → 調 c（即 DAMP_SCALE）
    alpha ：單一阻尼係數 d_i = α（傳統 PageRank）       → 調 α

回測方式（walk-forward，不偷看未來）：
    對球季中的每一天，只用「這一天之前」的比賽算 PageRank，再預測當天每一場：
        預測勝隊 = PageRank 較高的一方
        勝率     = PR(A) / (PR(A) + PR(B))
    兩隊都至少打過 MIN_GAMES 場才列入計分。

執行： python3 damping_backtest.py
"""
import csv
import json
import os
import time
import urllib.request

import numpy as np

SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_GAMES = 20                     # 兩隊都打滿這麼多場才開始預測
C_GRID = np.round(np.concatenate([np.arange(0.05, 1.0, 0.05),
                                  np.arange(1.0, 2.0, 0.1),
                                  np.arange(2.0, 6.01, 0.25)]), 2)
ALPHA_GRID = np.round(np.concatenate([np.arange(0.05, 0.96, 0.025), [0.975, 0.99]]), 3)

HERE = os.path.dirname(os.path.abspath(__file__))
API = 'https://statsapi.mlb.com/api/v1'


# ---------------------------------------------------------------- 資料
def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'python-urllib'})
    for attempt in range(4):                     # 整季賽程檔很大，偶爾會逾時，重試幾次
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5)


def fetch_games(season, path):
    """與 mlb_pagerank.m 的 fetch_games 相同格式的例行賽快取。"""
    abbr = {t['id']: t['abbreviation']
            for t in _get(f'{API}/teams?sportId=1&season={season}')['teams']}
    sj = _get(f'{API}/schedule?sportId=1&gameType=R'
              f'&startDate={season}-01-01&endDate={season}-12-31')
    rows = {}
    for d in sj['dates']:
        for g in d['games']:
            if g['status'].get('codedGameState') != 'F':
                continue
            a, h = g['teams']['away'], g['teams']['home']
            if a.get('score') is None or h.get('score') is None or a['score'] == h['score']:
                continue
            rows.setdefault(g['gamePk'], (
                g['gamePk'], g['officialDate'],
                a['team']['id'], abbr.get(a['team']['id'], a['team']['name'][:3].upper()), a['team']['name'],
                h['team']['id'], abbr.get(h['team']['id'], h['team']['name'][:3].upper()), h['team']['name'],
                a['score'], h['score']))
    out = sorted(rows.values(), key=lambda r: (r[1], r[0]))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamePk', 'date', 'away_id', 'away_abbr', 'away_name',
                    'home_id', 'home_abbr', 'home_name', 'away_score', 'home_score'])
        w.writerows(out)
    print(f'已寫入快取：{os.path.basename(path)}（{len(out)} 場）')


def fetch_postseason(season, path):
    """與 mlb_pagerank.m 的 fetch_postseason 相同格式的季後賽快取。"""
    sj = _get(f'{API}/schedule?sportId=1&gameType=F,D,L,W'
              f'&startDate={season}-01-01&endDate={season}-12-31')
    rows = {}
    for d in sj['dates']:
        for g in d['games']:
            if g['status'].get('codedGameState') != 'F':
                continue
            a, h = g['teams']['away'], g['teams']['home']
            if a.get('score') is None or h.get('score') is None or a['score'] == h['score']:
                continue
            rows.setdefault(g['gamePk'], (
                g['gamePk'], g['officialDate'], g['gameType'], g['seriesDescription'],
                g['gamesInSeries'], a['team']['id'], a['team']['name'],
                h['team']['id'], h['team']['name'], a['score'], h['score']))
    out = sorted(rows.values(), key=lambda r: (r[1], r[0]))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['gamePk', 'date', 'game_type', 'series', 'games_in_series',
                    'away_id', 'away_name', 'home_id', 'home_name', 'away_score', 'home_score'])
        w.writerows(out)
    print(f'已寫入季後賽快取：{os.path.basename(path)}（{len(out)} 場）')


def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- PageRank
def damping_sets(losses):
    """回傳 (K, n) 的阻尼值：前 len(C_GRID) 列是阻尼矩陣，其餘是單一 α。"""
    L = losses.astype(float)
    dm = L[None, :] / (L[None, :] + C_GRID[:, None] * L.mean())
    da = np.repeat(ALPHA_GRID[:, None], len(L), axis=1)
    return np.vstack([dm, da])


def pagerank_batch(A, d):
    """A：n×n 鄰接矩陣（輸家 → 贏家）；d：(K, n) 阻尼值。回傳 (K, n) 的 PageRank。"""
    n = A.shape[0]
    rs = A.sum(axis=1)
    P = np.full((n, n), 1.0 / n)
    ok = rs > 0
    P[ok] = A[ok] / rs[ok, None]
    G = d[:, :, None] * P[None] + (1 - d[:, :, None]) / n        # (K, n, n)
    M = np.transpose(G, (0, 2, 1)) - np.eye(n)[None]             # (G' - I) π = 0
    M[:, -1, :] = 1.0                                            # 換成 sum(π) = 1
    b = np.zeros((d.shape[0], n, 1))
    b[:, -1, 0] = 1.0
    return np.linalg.solve(M, b)[:, :, 0]


def adjacency(AH, AA):
    """主場勝 × 1/(2h)、客場勝 × 1/(2(1-h))，h = 目前為止的全聯盟主場勝率。"""
    nh, na = AH.sum(), AA.sum()
    h = nh / (nh + na)
    if h <= 0 or h >= 1:
        return AH + AA
    return AH / (2 * h) + AA / (2 * (1 - h))


# ---------------------------------------------------------------- 回測
def backtest_season(season):
    path = os.path.join(HERE, f'mlb_games_{season}.csv')
    if not os.path.exists(path):
        fetch_games(season, path)
    games = load_csv(path)
    ids = sorted({int(g['away_id']) for g in games} | {int(g['home_id']) for g in games})
    ix = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    K = len(C_GRID) + len(ALPHA_GRID)

    AH = np.zeros((n, n))            # 主場勝：AH[輸家, 贏家]
    AA = np.zeros((n, n))            # 客場勝
    W = np.zeros(n)
    L = np.zeros(n)

    hits = np.zeros(K)
    ll = np.zeros(K)
    brier = np.zeros(K)
    base = dict(n=0, pct=0.0, home=0.0, pct_ll=0.0)

    by_date = {}
    for g in games:
        by_date.setdefault(g['date'], []).append(g)

    for date in sorted(by_date):
        today = [(ix[int(g['away_id'])], ix[int(g['home_id'])],
                  int(g['away_score']) > int(g['home_score'])) for g in by_date[date]]
        played = W + L
        elig = [(a, h, aw) for a, h, aw in today if played[a] >= MIN_GAMES and played[h] >= MIN_GAMES]
        if elig:
            pr = pagerank_batch(adjacency(AH, AA), damping_sets(L))     # (K, n)
            pct = W / played.clip(min=1)
            for a, h, aw in elig:
                pa, ph = pr[:, a], pr[:, h]
                p_away = pa / (pa + ph)
                y = 1.0 if aw else 0.0
                # 平手（PR 相同）算 0.5 分
                hits += np.where(p_away == 0.5, 0.5, (p_away > 0.5) == aw)
                ll += -np.log(np.where(aw, p_away, 1 - p_away))
                brier += (p_away - y) ** 2
                # 對照組
                base['n'] += 1
                base['home'] += (not aw)
                x, z = pct[a], pct[h]
                q = x * (1 - z) / (x * (1 - z) + z * (1 - x)) if 0 < x < 1 and 0 < z < 1 else 0.5
                base['pct'] += 0.5 if q == 0.5 else float((q > 0.5) == aw)
                q = min(max(q, 1e-6), 1 - 1e-6)
                base['pct_ll'] += -np.log(q if aw else 1 - q)
        for a, h, aw in today:
            if aw:
                AA[h, a] += 1; W[a] += 1; L[h] += 1
            else:
                AH[a, h] += 1; W[h] += 1; L[a] += 1

    # 季後賽：用整季例行賽的 PageRank 預測每個系列賽的晉級隊
    ppath = os.path.join(HERE, f'mlb_postseason_{season}.csv')
    if not os.path.exists(ppath):
        fetch_postseason(season, ppath)
    pr = pagerank_batch(adjacency(AH, AA), damping_sets(L))
    series = {}
    for g in load_csv(ppath):
        a, h = ix[int(g['away_id'])], ix[int(g['home_id'])]
        key = (g['game_type'], min(a, h), max(a, h))
        s = series.setdefault(key, {'need': 0, 'w': {a: 0, h: 0}})
        s['need'] = max(s['need'], (int(g['games_in_series']) + 1) // 2)
        s['w'][a if int(g['away_score']) > int(g['home_score']) else h] += 1
    ps_hits = np.zeros(K)
    ps_n = 0
    for (_, a, b), s in series.items():
        win = [t for t, w in s['w'].items() if w == s['need']]
        if not win:
            continue
        ps_n += 1
        lose = b if win[0] == a else a
        ps_hits += np.where(pr[:, win[0]] == pr[:, lose], 0.5, pr[:, win[0]] > pr[:, lose])

    return dict(season=season, n=base['n'], hits=hits, ll=ll, brier=brier,
                base=base, ps_hits=ps_hits, ps_n=ps_n)


def main():
    res = [backtest_season(s) for s in SEASONS]
    nC = len(C_GRID)
    N = sum(r['n'] for r in res)
    hits = sum(r['hits'] for r in res)
    ll = sum(r['ll'] for r in res)
    brier = sum(r['brier'] for r in res)
    ps_hits = sum(r['ps_hits'] for r in res)
    ps_n = sum(r['ps_n'] for r in res)
    acc, mll, mbr = hits / N, ll / N, brier / N

    print(f'\n回測球季：{SEASONS}；計分場數 {N}（兩隊皆 ≥ {MIN_GAMES} 場後才預測）；季後賽系列賽 {ps_n} 個')
    print(f'對照組：主隊必勝 {sum(r["base"]["home"] for r in res) / N:.4f}，'
          f'勝率較高者勝 {sum(r["base"]["pct"] for r in res) / N:.4f}'
          f'（log5 log loss {sum(r["base"]["pct_ll"] for r in res) / N:.5f}），丟銅板 log loss {np.log(2):.5f}')

    out = os.path.join(HERE, 'damping_backtest_results.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'param', 'accuracy', 'log_loss', 'brier', 'postseason_series_hit_rate']
                   + [f'acc_{r["season"]}' for r in res] + [f'logloss_{r["season"]}' for r in res])
        for k in range(len(acc)):
            kind, val = ('matrix_c', C_GRID[k]) if k < nC else ('alpha', ALPHA_GRID[k - nC])
            w.writerow([kind, val, f'{acc[k]:.5f}', f'{mll[k]:.6f}', f'{mbr[k]:.6f}', f'{ps_hits[k] / ps_n:.4f}']
                       + [f'{r["hits"][k] / r["n"]:.5f}' for r in res]
                       + [f'{r["ll"][k] / r["n"]:.6f}' for r in res])
    print(f'完整結果已存檔：{out}')

    for name, sl, grid in (('阻尼矩陣 c（DAMP_SCALE）', slice(0, nC), C_GRID),
                           ('單一阻尼係數 α', slice(nC, None), ALPHA_GRID)):
        a, l, b, p = acc[sl], mll[sl], mbr[sl], ps_hits[sl] / ps_n
        print(f'\n=== {name} ===')
        print(f'{"param":>7} {"命中率":>8} {"log loss":>9} {"Brier":>8} {"季後賽":>7}')
        for k in range(len(grid)):
            mark = ('  ← 命中率最高' if k == a.argmax() else '') + ('  ← log loss 最低' if k == l.argmin() else '')
            print(f'{grid[k]:7.3f} {a[k]:8.4f} {l[k]:9.5f} {b[k]:8.5f} {p[k]:7.3f}{mark}')
        print('各球季各自的最佳值（看穩不穩定）：')
        for r in res:
            ra, rl = r['hits'][sl] / r['n'], r['ll'][sl] / r['n']
            print(f'    {r["season"]}：命中率最高 {grid[ra.argmax()]:.3f}（{ra.max():.4f}），'
                  f'log loss 最低 {grid[rl.argmin()]:.3f}（{rl.min():.5f}）')
        # 留一球季交叉驗證：用其他 4 季挑參數，在剩下那一季驗證
        cv_a = cv_l = 0.0
        for r in res:
            oth_h = hits[sl] - r['hits'][sl]
            oth_l = ll[sl] - r['ll'][sl]
            cv_a += r['hits'][sl][oth_h.argmax()]
            cv_l += r['ll'][sl][oth_l.argmin()]
        print(f'留一球季交叉驗證：依命中率挑 → 樣本外命中率 {cv_a / N:.4f}；'
              f'依 log loss 挑 → 樣本外 log loss {cv_l / N:.5f}')


if __name__ == '__main__':
    main()
