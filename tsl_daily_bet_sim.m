function tsl_daily_bet_sim(season, stake, bank0)
% TSL_DAILY_BET_SIM  單季逐日下注模擬：分段阻尼 PageRank，比較多種下注方法
%
%   tsl_daily_bet_sim                     % 2026 球季，每注 100 元，凱利起始資金 10000 元
%   tsl_daily_bet_sim(2016, 100, 10000)   % 換球季驗證
%
%   模型（逐日改變阻尼）
%     每天只用「今天以前」的本季比賽建網絡。每場比賽依兩隊中較少的已賽場數 G，
%     查 damping_by_games.csv（2021–2025 回測得到、每 10 場一組的最佳阻尼）決定 α，
%     用經典 PageRank（均勻跳轉）G = α·P + (1−α)/n 算 π，pRaw = π_主 / (π_主 + π_客)。
%     G < 10 不下注；G ≥ 160 沿用最後一組的 α。
%   主場校準
%     分段阻尼模型本身沒有主場優勢，會系統性高估客隊。這裡用「其他球季」
%     （2016–2025，扣掉 season）擬合 p = σ(a·logit(pRaw) + b)，不用到當季資料。
%     也會列出未校準的結果做對照。
%   下注
%     賠率來自 espn_odds_YYYY.csv（tsl_fetch_espn_odds 下載；2026 只有 DraftKings）。
%     每場算兩邊 EV = p × 賠率 − 1。比較的方法：
%       原方法          EV 較大的一邊若 > 0 就押 stake 元
%       限制賠率        同上，但只押賠率 < 2.0（或 < 1.8）的一邊，避開冷門
%       只押熱門        同上，但只押賠率較低（市場看好）的一邊
%       1/4 凱利        金額 = 當時資金 × 凱利比例 / 4（最少 stake 元），可再加賠率 < 2.0
%       對照            每場押主隊、每場押熱門（不看模型）
%
%   輸出：比較表、每月損益、累積損益圖（tsl_daily_bet_<season>.png）、
%         每日累積損益（tsl_daily_bet_<season>.csv）

    if nargin < 1 || isempty(season), season = 2026; end
    if nargin < 2 || isempty(stake),  stake  = 100;  end
    if nargin < 3 || isempty(bank0),  bank0  = 10000; end
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    MIN_GAMES = 10;

    tab = read_damping_table(fullfile(here, 'damping_by_games.csv'));
    fprintf('分段阻尼（依兩隊中較少的已賽場數）：\n');
    for i = 1:numel(tab.lo)
        fprintf('  %3d–%3d 場：α = %g\n', tab.lo(i), tab.hi(i), tab.alpha(i));
    end

    % ------------------------------------------------------------------
    %  1. 主場校準：用其他球季擬合 a、b
    % ------------------------------------------------------------------
    train = setdiff(2016:2025, season);
    fprintf('\n用 %d–%d（不含 %d）擬合主場校準…\n', train(1), train(end), season);
    Xtr = [];  Ytr = [];
    for s = train
        [x, yy] = piecewise_probs(here, s, tab, MIN_GAMES);
        Xtr = [Xtr; x];  Ytr = [Ytr; yy];                   %#ok<AGROW>
    end
    w = tsl_fit_logistic(Xtr, Ytr);
    fprintf('  a = %.3f，b = %+.3f（實力相當時主隊勝率 %.3f）\n', w(1), w(2), tsl_calibrate(0.5, w(1), w(2)));

    % ------------------------------------------------------------------
    %  2. 目標球季：模型勝率對上賠率
    % ------------------------------------------------------------------
    [x, y, info] = piecewise_probs(here, season, tab, MIN_GAMES);
    od  = read_odds(fullfile(here, sprintf('espn_odds_%d.csv', season)));
    idx = match_odds(info, od);
    ok  = idx > 0;  ok(ok) = od.n(idx(ok)) > 0;
    j   = idx(ok);
    x = x(ok);  y = y(ok);  day = info.date(ok);
    oA = od.aMed(j);  oH = od.hMed(j);
    fprintf('\n%d：模型可預測 %d 場，其中 %d 場有賠率；平均抽水 %.2f%%\n', season, numel(ok), sum(ok), ...
        100 * mean(od.over(j)));
    pCal = tsl_calibrate(x, w(1), w(2));
    fprintf('  log loss：校準後 %.5f｜未校準 %.5f｜市場 %.5f｜丟銅板 %.5f\n', tsl_logloss(pCal, y), ...
        tsl_logloss(x, y), tsl_logloss(od.pHome(j), y), log(2));

    % ------------------------------------------------------------------
    %  3. 各種下注方法（都用校準後的勝率）
    % ------------------------------------------------------------------
    %   mode   'ev'   每場挑 EV 較大的一邊，EV > 0 才押
    %          'home' 每場都押主隊（對照）
    %          'fav'  每場都押熱門＝賠率較低的一邊（對照）
    %   maxOdds 只押賠率低於此值的一邊；favOnly 只押熱門；kelly 依凱利公式決定金額
    S = struct('name', {}, 'mode', {}, 'maxOdds', {}, 'favOnly', {}, 'kelly', {});
    S(end + 1) = strat('原方法：正EV、固定金額',      'ev',   inf, false, false);
    S(end + 1) = strat('正EV、賠率 < 2.0',            'ev',   2.0, false, false);
    S(end + 1) = strat('正EV、賠率 < 1.8',            'ev',   1.8, false, false);
    S(end + 1) = strat('正EV、只押熱門',              'ev',   inf, true,  false);
    S(end + 1) = strat('正EV、1/4 凱利',              'ev',   inf, false, true);
    S(end + 1) = strat('正EV、1/4 凱利、賠率 < 2.0',  'ev',   2.0, false, true);
    S(end + 1) = strat('對照：每場押主隊',            'home', inf, false, false);
    S(end + 1) = strat('對照：每場押熱門',            'fav',  inf, false, false);

    [days, ~, di] = unique(day);
    nD = numel(days);
    fprintf('\n下注方法比較（固定金額每注 %d 元；凱利起始資金 %d 元、每注最少 %d 元）\n', stake, bank0, stake);
    fprintf('  %-26s %5s %9s %9s %8s %19s %6s %6s %8s\n', '方法', '注數', '投注金額', '損益', 'ROI', ...
        '95% 信賴區間', '押中率', '平均賠率', '最大回撤');
    R = cell(numel(S), 1);
    for k = 1:numel(S)
        r = run_strategy(S(k), di, nD, pCal, y, oA, oH, stake, bank0);
        R{k} = r;
        fprintf('  %-26s %5d %9.0f %+9.0f %+7.2f%% [%+6.2f%%, %+6.2f%%] %5.1f%% %6.2f %8.0f\n', S(k).name, ...
            r.n, r.staked, r.profit, 100 * r.roi, 100 * r.ci(1), 100 * r.ci(2), 100 * r.hit, r.avgOdds, r.maxDD);
    end
    for k = find([S.kelly])
        fprintf('  %s：期末資金 %.0f 元（起始 %d 元）\n', S(k).name, bank0 + R{k}.profit, bank0);
    end

    % --- 每月損益 ---
    mon = cellfun(@(s) s(1:7), days, 'UniformOutput', false);
    [um, ~, mi] = unique(mon);
    fprintf('\n每月損益（元）\n  %-8s', '月份');
    fprintf(' %10s', S(1:6).name);  fprintf('\n');
    for m = 1:numel(um)
        fprintf('  %-8s', um{m});
        for k = 1:6
            fprintf(' %+10.0f', sum(R{k}.daily(mi == m)));
        end
        fprintf('\n');
    end

    % --- 存檔：每日累積損益、圖 ---
    out = fullfile(here, sprintf('tsl_daily_bet_%d.csv', season));
    fid = fopen(out, 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', char(65279));
    fprintf(fid, 'date');  fprintf(fid, ',%s', S.name);  fprintf(fid, '\n');
    for i = 1:nD
        fprintf(fid, '%s', days{i});
        for k = 1:numel(S), fprintf(fid, ',%.2f', R{k}.cum(i)); end
        fprintf(fid, '\n');
    end
    fclose(fid);

    fig = figure('Visible', 'off', 'Position', [100 100 1000 480]);
    t = datenum(days, 'yyyy-mm-dd');
    hold on;
    for k = 1:numel(S)
        if k > 6, ls = ':'; else, ls = '-'; end
        plot(t, R{k}.cum, ls, 'LineWidth', 1.3);
    end
    plot(t([1 end]), [0 0], 'k:');
    datetick('x', 'mm/dd');  grid on;
    xlabel('日期');  ylabel('累積損益（元）');
    legend({S.name}, 'Location', 'eastoutside');
    title(sprintf('%d 分段阻尼 PageRank：各種下注方法的累積損益', season));
    png = fullfile(here, sprintf('tsl_daily_bet_%d.png', season));
    print(fig, png, '-dpng', '-r120');  close(fig);
    fprintf('\n已存檔：%s\n        %s\n', out, png);
end

% ======================================================================
%  區域函式
% ======================================================================
function s = strat(name, mode, maxOdds, favOnly, kelly)
    s = struct('name', name, 'mode', mode, 'maxOdds', maxOdds, 'favOnly', favOnly, 'kelly', kelly);
end

function r = run_strategy(S, di, nD, p, y, oA, oH, stake, bank0)
% 逐日模擬一種下注方法。凱利：每天開賽前依當時資金 B 決定每注金額
%   f = (p·o − 1) / (o − 1)（完整凱利比例），押 B·f/4，四捨五入到 10 元，最少 stake 元；
%   當天總投注不超過 B；資金低於 stake 就停止下注。當天比賽結束後結算。
    evH = p .* oH - 1;   evA = (1 - p) .* oA - 1;
    switch S.mode
        case 'ev',   home = evH >= evA;  take = max(evH, evA) > 0;
        case 'home', home = true(size(p));  take = true(size(p));
        case 'fav',  home = oH <= oA;       take = true(size(p));
    end
    od    = oA;  od(home)    = oH(home);           % 押的那一邊的賠率
    other = oH;  other(home) = oA(home);           % 對手的賠率
    pS    = 1 - p;  pS(home) = p(home);            % 模型認為押的那一邊會贏的機率
    take  = take & od < S.maxOdds;
    if S.favOnly, take = take & od <= other; end
    win   = (home & y == 1) | (~home & y == 0);

    amt = zeros(size(p));  B = bank0;
    for d = 1:nD
        g = find(di == d & take);
        if isempty(g), continue; end
        if S.kelly
            if B < stake, continue; end
            f = (pS(g) .* od(g) - 1) ./ (od(g) - 1);
            a = max(stake, round(B * f / 4 / 10) * 10);
            if sum(a) > B, a = floor(a * (B / sum(a)) / 10) * 10; end
        else
            a = stake * ones(size(g));
        end
        amt(g) = a;
        B = B + sum(a .* (win(g) .* (od(g) - 1) - ~win(g)));
    end
    pnl = amt .* (win .* (od - 1) - ~win);
    b   = amt > 0;
    r.n = sum(b);  r.staked = sum(amt);  r.profit = sum(pnl);
    r.roi = r.profit / max(r.staked, 1);
    rr  = pnl(b) ./ amt(b);                        % 每注報酬率
    h   = 1.96 * std(rr) / sqrt(max(r.n, 1));
    r.ci = [mean(rr) - h, mean(rr) + h];
    r.hit = mean(win(b));  r.avgOdds = mean(od(b));
    r.daily = accumarray(di, pnl, [nD 1]);
    r.cum = cumsum(r.daily);
    r.maxDD = max(cummax([0; r.cum]) - [0; r.cum]);
end

function [X, y, info] = piecewise_probs(here, season, tab, minGames)
% 逐日滾動：每場依兩隊中較少的已賽場數查表決定 α，算經典 PageRank 的主隊原始勝率
    f = fullfile(here, sprintf('mlb_games_%d.csv', season));
    if exist(f, 'file') ~= 2, tsl_fetch_games(season, f); end
    g = tsl_read_games(f);
    ids = unique([g.awayId; g.homeId]);  n = numel(ids);
    [~, ai] = ismember(g.awayId, ids);
    [~, hi] = ismember(g.homeId, ids);
    awayWon = g.awayScore > g.homeScore;
    [~, ~, di] = unique(g.date);
    AH = zeros(n);  AA = zeros(n);  W = zeros(n, 1);  L = zeros(n, 1);
    X = zeros(numel(ai), 1);  y = X;  keep = X;  m = 0;
    for d = 1:max(di)
        today  = find(di == d);
        played = W + L;
        G      = min(played(ai(today)), played(hi(today)));
        elig   = today(G >= minGames);
        if ~isempty(elig)
            a = lookup_alpha(tab, min(played(ai(elig)), played(hi(elig))));
            [ua, ~, ui] = unique(a);
            pr = classic_pagerank(AH, AA, ua);              % numel(ua) × n
            for k = 1:numel(elig)
                t = elig(k);  r = ui(k);
                m = m + 1;
                X(m) = pr(r, hi(t)) / (pr(r, hi(t)) + pr(r, ai(t)));
                y(m) = ~awayWon(t);  keep(m) = t;
            end
        end
        for t = today'
            a = ai(t);  h = hi(t);
            if awayWon(t)
                AA(h, a) = AA(h, a) + 1;  W(a) = W(a) + 1;  L(h) = L(h) + 1;
            else
                AH(a, h) = AH(a, h) + 1;  W(h) = W(h) + 1;  L(a) = L(a) + 1;
            end
        end
    end
    X = X(1:m);  y = y(1:m);  keep = keep(1:m);
    info.date = g.date(keep);  info.awayId = g.awayId(keep);  info.homeId = g.homeId(keep);
    info.awayScore = g.awayScore(keep);  info.homeScore = g.homeScore(keep);
end

function a = lookup_alpha(tab, G)
% 已賽場數 G → 該組的最佳 α（超過表的範圍就用最後一組）
    a = zeros(size(G));
    for k = 1:numel(G)
        i = find(G(k) >= tab.lo & G(k) < tab.hi, 1);
        if isempty(i), i = numel(tab.lo); end
        a(k) = tab.alpha(i);
    end
end

function pr = classic_pagerank(AH, AA, alphas)
% 經典 PageRank（均勻跳轉）：G = α·P + (1−α)/n，邊權重同樣做主客場反機率加權
    n  = size(AH, 1);
    nh = sum(AH(:));  na = sum(AA(:));  h = nh / max(nh + na, eps);
    if h <= 0 || h >= 1, A = AH + AA; else, A = AH / (2 * h) + AA / (2 * (1 - h)); end
    rs = sum(A, 2);  P = ones(n) / n;  ok = rs > 0;
    P(ok, :) = A(ok, :) ./ repmat(rs(ok), 1, n);
    pr  = zeros(numel(alphas), n);
    rhs = zeros(n, 1);  rhs(end) = 1;
    for k = 1:numel(alphas)
        M = (alphas(k) * P + (1 - alphas(k)) / n)' - eye(n);
        M(end, :) = 1;
        pr(k, :) = (M \ rhs)';
    end
end

function tab = read_damping_table(f)
% 讀 damping_by_games.csv：games_from, games_to, n_games, best_alpha, ...
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s', f); end
    C = textscan(fid, '%f %f %f %f %*[^\n]', 'Delimiter', ',', 'HeaderLines', 1);
    fclose(fid);
    tab.lo = C{1};  tab.hi = C{2};  tab.alpha = C{4};
end

function idx = match_odds(info, od)
% 用「日期｜客隊｜主隊｜客隊得分｜主隊得分」對應；找不到再試前後一天
    key = @(d, a, h, as, hs) sprintf('%s|%d|%d|%d|%d', d, a, h, as, hs);
    M = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for j = 1:numel(od.n)
        M(key(od.date{j}, od.awayId(j), od.homeId(j), od.awayScore(j), od.homeScore(j))) = j;
    end
    idx = zeros(numel(info.date), 1);
    for i = 1:numel(info.date)
        d0 = datenum(info.date{i}, 'yyyy-mm-dd');
        for dd = [0 -1 1]
            k = key(datestr(d0 + dd, 'yyyy-mm-dd'), info.awayId(i), info.homeId(i), ...
                    info.awayScore(i), info.homeScore(i));
            if isKey(M, k), idx(i) = M(k); break; end
        end
    end
end

function od = read_odds(f)
% 讀 espn_odds_YYYY.csv
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s，請先執行 tsl_fetch_espn_odds。', f); end
    C = textscan(fid, '%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    od.date = C{2};  od.awayId = C{3};  od.homeId = C{4};
    od.awayScore = C{5};  od.homeScore = C{6};  od.n = C{7};
    od.aMed = C{8};  od.hMed = C{9};  od.pHome = C{12};  od.over = C{13};
end
