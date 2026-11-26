function tsl_pitcher_trial(day, nGames, halfLife)
% TSL_PITCHER_TRIAL  小規模試算：加入「先發投手」與「近期比賽加權」後，勝率怎麼變
%
%   tsl_pitcher_trial                         % 2026-08-15 的前 10 場，半衰期 30 天
%   tsl_pitcher_trial('2026-07-01', 10, 20)
%
%   比較四種模型（都只用 day 以前的資料，校準參數讀自 tsl_calibration.csv）
%     基本      依勝率跳轉 PageRank（α 同 tsl_calibration.csv）＋主場校準
%     近期加權  每場比賽的權重 = 0.5^(距今天數 / halfLife)，邊權重與勝敗場數都加權
%     先發投手  基本模型的 logit 再加上 K × (客隊先發 ERA − 主隊先發 ERA)
%     兩者      近期加權＋先發投手
%
%   先發投手
%     名單：MLB Stats API 賽程的 probablePitcher（賽前就公布的預定先發，不會偷看結果）
%     能力：該投手本季到前一天為止的 ERA，向聯盟平均收縮：
%           ERA* = 9 × (自責分 + 聯盟ERA × IP0/9) / (局數 + IP0)，IP0 = 30 局
%     換成勝率：先發約投 5.5 局，用畢氏勝率（指數 1.83、每隊每場約 4.4 分）近似
%           Δlogit(主隊) = 1.83 × (5.5/9) × (ERA*_客 − ERA*_主) / 4.4 ≈ 0.254 × ERA 差
%     這是理論近似，K 沒有用資料擬合；PageRank 本身已含球隊的平均投手實力，
%     所以這個調整可能有一部分重複計算。
%
%   注意：只有 10 場，log loss 差異完全不具統計意義，只能看「方向對不對」。

    if nargin < 1 || isempty(day),      day = '2026-08-15'; end
    if nargin < 2 || isempty(nGames),   nGames = 10;        end
    if nargin < 3 || isempty(halfLife), halfLife = 30;      end
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    LG_ERA = 4.20;  IP0 = 30;  K = 1.83 * (5.5 / 9) / 4.4;
    season = str2double(day(1:4));
    opts = weboptions('Timeout', 60, 'ContentType', 'text');

    cal = read_calibration(fullfile(here, 'tsl_calibration.csv'));
    g   = tsl_read_games(fullfile(here, sprintf('mlb_games_%d.csv', season)));
    ids = unique([g.awayId; g.homeId]);  n = numel(ids);
    [~, ai] = ismember(g.awayId, ids);
    [~, hi] = ismember(g.homeId, ids);
    gd  = datenum(g.date, 'yyyy-mm-dd');
    d0  = datenum(day, 'yyyy-mm-dd');

    % ------------------------------------------------------------------
    %  1. PageRank：不加權 vs 近期加權
    % ------------------------------------------------------------------
    past = find(gd < d0);
    wAll = ones(numel(past), 1);
    wRec = 0.5 .^ ((d0 - gd(past)) / halfLife);
    prB  = rank_with_weights(past, wAll, ai, hi, g, n, cal.alpha);
    prR  = rank_with_weights(past, wRec, ai, hi, g, n, cal.alpha);

    % ------------------------------------------------------------------
    %  2. 當天比賽與預定先發
    % ------------------------------------------------------------------
    sj = jsondecode(webread(sprintf(['https://statsapi.mlb.com/api/v1/schedule?sportId=1' ...
        '&date=%s&hydrate=probablePitcher'], day), opts));
    d1 = elem(sj.dates, 1);  sg = d1.games;
    today = find(gd == d0);
    today = today(1:min(nGames, numel(today)));
    od = read_odds(fullfile(here, sprintf('espn_odds_%d.csv', season)));

    fprintf('%s 的 %d 場（PageRank α = %g，近期加權半衰期 %d 天，K = %.3f）\n\n', ...
        day, numel(today), cal.alpha, halfLife, K);
    fprintf('%-26s %-30s %6s %6s %6s %6s %6s  %s\n', '比賽（客 @ 主）', '先發 ERA*（客 / 主）', ...
        '市場', '基本', '近期', '先發', '兩者', '結果');
    P = zeros(numel(today), 5);  Y = zeros(numel(today), 1);
    for k = 1:numel(today)
        t = today(k);
        s = find_game(sg, g.gamePk(t));
        [nmA, eA] = pitcher_era(s, 'away', d0, LG_ERA, IP0, opts);
        [nmH, eH] = pitcher_era(s, 'home', d0, LG_ERA, IP0, opts);
        dl = K * (eA - eH);                                % 主隊的 logit 調整

        pB = tsl_calibrate(prB(hi(t)) / (prB(hi(t)) + prB(ai(t))), cal.a, cal.b);
        pR = tsl_calibrate(prR(hi(t)) / (prR(hi(t)) + prR(ai(t))), cal.a, cal.b);
        pS = sigm(logit(pB) + dl);
        p2 = sigm(logit(pR) + dl);
        pM = market_prob(od, g, t);
        Y(k) = g.homeScore(t) > g.awayScore(t);
        P(k, :) = [pM pB pR pS p2];
        if Y(k), res = sprintf('%s 勝 %d-%d', g.homeAbbr{t}, g.homeScore(t), g.awayScore(t));
        else,    res = sprintf('%s 勝 %d-%d', g.awayAbbr{t}, g.awayScore(t), g.homeScore(t)); end
        fprintf('%-26s %-30s %5.1f%% %5.1f%% %5.1f%% %5.1f%% %5.1f%%  %s\n', ...
            sprintf('%s @ %s', g.awayAbbr{t}, g.homeAbbr{t}), ...
            sprintf('%s %.2f / %s %.2f', short(nmA), eA, short(nmH), eH), 100 * P(k, :), res);
    end
    fprintf('（表中百分比都是「主隊勝率」）\n\n');

    names = {'市場', '基本', '近期加權', '先發投手', '兩者'};
    fprintf('%-10s %9s %16s\n', '模型', 'log loss', '與市場平均差距');
    for j = 1:5
        ok = ~isnan(P(:, j));
        fprintf('%-10s %9.4f %15.1f 點\n', names{j}, tsl_logloss(P(ok, j), Y(ok)), ...
            100 * mean(abs(P(ok, j) - P(ok, 1)), 'omitnan'));
    end
    fprintf('\n只有 %d 場，以上差異不具統計意義；要判斷有沒有用，需要跑整季甚至多季。\n', numel(today));
end

% ======================================================================
%  區域函式
% ======================================================================
function pr = rank_with_weights(idx, w, ai, hi, g, n, alpha)
% 用權重 w 建網絡（AH、AA、勝敗場數都乘上權重），算依勝率跳轉的 PageRank
    AH = zeros(n);  AA = zeros(n);  W = zeros(n, 1);  L = zeros(n, 1);
    for k = 1:numel(idx)
        t = idx(k);  a = ai(t);  h = hi(t);  x = w(k);
        if g.awayScore(t) > g.homeScore(t)
            AA(h, a) = AA(h, a) + x;  W(a) = W(a) + x;  L(h) = L(h) + x;
        else
            AH(a, h) = AH(a, h) + x;  W(h) = W(h) + x;  L(a) = L(a) + x;
        end
    end
    pr = tsl_pagerank(AH, AA, W, L, alpha);
end

function s = find_game(sg, pk)
% 在當天賽程中找 gamePk 對應的比賽
    s = [];
    for i = 1:numel(sg)
        x = elem(sg, i);
        if x.gamePk == pk, s = x; return; end
    end
end

function [name, era] = pitcher_era(s, side, d0, lgEra, ip0, opts)
% 預定先發投手本季到前一天為止的收縮 ERA；查不到就用聯盟平均
    name = '?';  era = lgEra;
    if isempty(s), return; end
    tm = s.teams.(side);
    if ~isfield(tm, 'probablePitcher') || isempty(tm.probablePitcher), return; end
    pp = tm.probablePitcher;  name = pp.fullName;
    yr = str2double(datestr(d0, 'yyyy'));
    js = jsondecode(webread(sprintf(['https://statsapi.mlb.com/api/v1/people/%d/stats' ...
        '?stats=gameLog&season=%d&group=pitching'], pp.id, yr), opts));
    if isempty(js.stats), return; end
    st = elem(js.stats, 1);                       % MATLAB 不能直接對函式回傳值取欄位
    if ~isfield(st, 'splits') || isempty(st.splits), return; end
    sp = st.splits;
    ip = 0;  er = 0;
    for i = 1:numel(sp)
        x = elem(sp, i);
        if datenum(x.date, 'yyyy-mm-dd') >= d0, continue; end
        ip = ip + innings(x.stat.inningsPitched);
        er = er + x.stat.earnedRuns;
    end
    era = 9 * (er + lgEra * ip0 / 9) / (ip + ip0);
end

function v = innings(s)
% "5.2" 代表 5 又 2/3 局
    if isnumeric(s), s = sprintf('%.1f', s); end
    p = sscanf(s, '%d.%d');
    v = p(1);  if numel(p) > 1, v = v + p(2) / 3; end
end

function p = market_prob(od, g, t)
% 對應 espn_odds 的市場主隊勝率（日期＋兩隊＋比分）
    m = strcmp(od.date, g.date{t}) & od.awayId == g.awayId(t) & od.homeId == g.homeId(t) ...
        & od.awayScore == g.awayScore(t) & od.homeScore == g.homeScore(t);
    k = find(m, 1);
    if isempty(k), p = NaN; else, p = od.pHome(k); end
end

function od = read_odds(f)
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s', f); end
    C = textscan(fid, '%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    od.date = C{2};  od.awayId = C{3};  od.homeId = C{4};
    od.awayScore = C{5};  od.homeScore = C{6};  od.pHome = C{12};
end

function cal = read_calibration(f)
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s', f); end
    fgetl(fid);  c = strsplit(fgetl(fid), ',');  fclose(fid);
    cal.alpha = str2double(c{1});  cal.a = str2double(c{2});  cal.b = str2double(c{3});
end

function s = short(name)
% 只留姓氏，表格比較好排
    w = strsplit(name, ' ');  s = w{end};
end

function x = elem(arr, i)
    if iscell(arr), x = arr{i}; else, x = arr(i); end
end

function z = logit(p), z = log(p ./ (1 - p)); end
function p = sigm(z),  p = 1 ./ (1 + exp(-z)); end
