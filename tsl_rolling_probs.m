function [X, y, info] = tsl_rolling_probs(here, season, alphas, minGames, halfLife)
% TSL_ROLLING_PROBS  逐日滾動計算某一季每場比賽的主隊原始勝率（各 α 一欄）
%
%   [X, y, info] = tsl_rolling_probs(here, season, alphas, minGames)
%   [X, y, info] = tsl_rolling_probs(here, season, alphas, minGames, halfLife)
%
%   halfLife（選填，單位：天）：近期比賽加權，每場權重 = 0.5^(距今天數 / halfLife)，
%   邊權重與勝敗場數都加權；省略或 inf 代表不加權（原本的做法）。
%
%   每天先用「今天以前」的比賽算依勝率跳轉的 PageRank、預測今天的比賽，
%   再把今天的結果加進網絡。兩隊都打滿 minGames 場才收錄。
%
%   輸出
%     X     m × numel(alphas)，主隊原始勝率 π_主 / (π_主 + π_客)
%     y     m × 1，主隊是否獲勝（1 / 0）
%     info  struct，欄位 gamePk、date（cellstr）、awayId、homeId、awayScore、homeScore，
%           用來和賠率資料對應
%
%   需要 mlb_games_YYYY.csv 快取（沒有的話會自動從 MLB Stats API 下載）。
    if nargin < 5 || isempty(halfLife), halfLife = inf; end
    f = fullfile(here, sprintf('mlb_games_%d.csv', season));
    if exist(f, 'file') ~= 2
        fprintf('  沒有 %s，從 MLB Stats API 下載…\n', f);
        tsl_fetch_games(season, f);
    end
    g = tsl_read_games(f);
    ids = unique([g.awayId; g.homeId]);
    n   = numel(ids);
    [~, ai] = ismember(g.awayId, ids);
    [~, hi] = ismember(g.homeId, ids);
    awayWon = g.awayScore > g.homeScore;
    [ud, ~, di] = unique(g.date);                     % 每場比賽屬於第幾個比賽日（已排序）
    udn = datenum(ud, 'yyyy-mm-dd');

    AH = zeros(n);  AA = zeros(n);  W = zeros(n, 1);  L = zeros(n, 1);
    X = zeros(numel(ai), numel(alphas));  y = zeros(numel(ai), 1);
    keep = zeros(numel(ai), 1);  m = 0;
    for d = 1:max(di)
        today  = find(di == d);
        played = W + L;
        elig   = today(min(played(ai(today)), played(hi(today))) >= minGames);
        if ~isempty(elig)
            if isinf(halfLife)
                pr = tsl_pagerank(AH, AA, W, L, alphas);   % numel(alphas) × n
            else                                           % 近期加權：每天依距今天數重建網絡
                [wAH, wAA, wW, wL] = weighted_net(find(di < d), udn(di(di < d)), udn(d), ...
                    halfLife, ai, hi, awayWon, n);
                pr = tsl_pagerank(wAH, wAA, wW, wL, alphas);
            end
            for t = elig'
                m = m + 1;
                X(m, :)  = (pr(:, hi(t)) ./ (pr(:, hi(t)) + pr(:, ai(t))))';
                y(m)     = ~awayWon(t);
                keep(m)  = t;
            end
        end
        for t = today'                                % 把今天的結果加進網絡
            a = ai(t);  h = hi(t);
            if awayWon(t)
                AA(h, a) = AA(h, a) + 1;  W(a) = W(a) + 1;  L(h) = L(h) + 1;
            else
                AH(a, h) = AH(a, h) + 1;  W(h) = W(h) + 1;  L(a) = L(a) + 1;
            end
        end
    end
    X = X(1:m, :);  y = y(1:m);  keep = keep(1:m);
    info.gamePk    = g.gamePk(keep);
    info.date      = g.date(keep);
    info.awayId    = g.awayId(keep);
    info.homeId    = g.homeId(keep);
    info.awayScore = g.awayScore(keep);
    info.homeScore = g.homeScore(keep);
end

function [AH, AA, W, L] = weighted_net(idx, dates, today, halfLife, ai, hi, awayWon, n)
% 以 0.5^(距今天數 / halfLife) 為權重，重建 idx 這些比賽的網絡
    AH = zeros(n);  AA = zeros(n);  W = zeros(n, 1);  L = zeros(n, 1);
    w  = 0.5 .^ ((today - dates) / halfLife);
    for k = 1:numel(idx)
        t = idx(k);  a = ai(t);  h = hi(t);  x = w(k);
        if awayWon(t)
            AA(h, a) = AA(h, a) + x;  W(a) = W(a) + x;  L(h) = L(h) + x;
        else
            AH(a, h) = AH(a, h) + x;  W(h) = W(h) + x;  L(a) = L(a) + x;
        end
    end
end
