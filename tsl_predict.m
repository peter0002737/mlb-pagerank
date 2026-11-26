function tsl_predict(oddsFile, refresh, minEV)
% TSL_PREDICT  運彩 MLB「不讓分」預測：PageRank 勝率 vs 運彩賠率，算出每一邊的期望值
%
%   流程
%     1. 用 tsl_odds_snippet.js 在瀏覽器下載當天賠率 tsl_odds_YYYY-MM-DD.csv
%     2. 在 MATLAB 執行：
%          tsl_predict                          % 自動找專案資料夾與「下載」資料夾中最新的 tsl_odds_*.csv
%          tsl_predict('tsl_odds_2027-04-01.csv')
%          tsl_predict(檔案, false)             % 不重新下載本季賽程，直接用快取
%          tsl_predict(檔案, true, 0.05)        % 只標示 EV > 5% 的選項
%
%   模型（見 tsl_calibration_backtest）
%     依勝率跳轉的 PageRank（只用開賽日以前的本季比賽）
%     → pRaw = π_主 / (π_主 + π_客)
%     → 校準 p_主 = σ(a·logit(pRaw) + b)，α、a、b 讀自 tsl_calibration.csv
%
%   輸出
%     模型勝率、運彩隱含勝率（去掉抽水後）、期望值 EV = p × 賠率 − 1（每投注 1 元的平均盈虧）
%     存成 tsl_predictions_YYYY-MM-DD.csv
%
%   注意：模型只看勝負，不知道先發投手、傷兵、休息日；運彩抽水約 15–17%。
%   回測中模型最有把握（≥ 65%）的比賽實際勝率約 0.67，要賠率 ≥ 1.50 才不賠錢。
%   EV 只是模型的估計，不是保證。

    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    if nargin < 1 || isempty(oddsFile), oddsFile = find_odds_file(here); end
    if nargin < 2 || isempty(refresh),  refresh  = true; end
    if nargin < 3 || isempty(minEV),    minEV    = 0;    end
    MIN_GAMES = 10;                                    % 兩隊都打滿這麼多場才預測

    % ------------------------------------------------------------------
    %  1. 讀賠率與校準參數
    % ------------------------------------------------------------------
    if isempty(oddsFile) || exist(oddsFile, 'file') ~= 2
        error(['找不到賠率檔。請先用 tsl_odds_snippet.js 下載 tsl_odds_YYYY-MM-DD.csv，' ...
               '或用 tsl_predict(''檔案路徑'') 指定。']);
    end
    odds = read_odds(oddsFile);
    if isempty(odds.away), error('%s 沒有任何賽事。', oddsFile); end
    fprintf('賠率檔：%s（%d 場）\n', oddsFile, numel(odds.away));

    cal = read_calibration(fullfile(here, 'tsl_calibration.csv'));

    % ------------------------------------------------------------------
    %  2. 本季賽程（只用開賽日以前的比賽）
    % ------------------------------------------------------------------
    gameDay = cellfun(@us_date, odds.start);           % 每場比賽的美東日期（datenum）
    dv      = datevec(min(gameDay));
    season  = dv(1);
    now_    = clock;
    gpath   = fullfile(here, sprintf('mlb_games_%d.csv', season));
    if exist(gpath, 'file') ~= 2 || (refresh && season >= now_(1))
        fprintf('下載 %d 賽程…\n', season);
        tsl_fetch_games(season, gpath);
    end
    g     = tsl_read_games(gpath);
    gDay  = datenum(g.date, 'yyyy-mm-dd');
    ids   = unique([g.awayId; g.homeId]);
    n     = numel(ids);
    [~, ai] = ismember(g.awayId, ids);
    [~, hi] = ismember(g.homeId, ids);
    awayWon = g.awayScore > g.homeScore;
    engName = cell(n, 1);                               % 英文隊名（隊名對應的備援）
    engName(ai) = g.awayName;  engName(hi) = g.homeName;

    % ------------------------------------------------------------------
    %  3. 逐場計算模型勝率與期望值
    % ------------------------------------------------------------------
    R = struct('start', {}, 'game', {}, 'side', {}, 'team', {}, 'record', {}, ...
               'p', {}, 'odds', {}, 'implied', {}, 'ev', {}, 'over', {});
    for i = 1:numel(odds.away)
        ta = team_index(odds.away{i}, ids, engName);
        th = team_index(odds.home{i}, ids, engName);
        if ta == 0 || th == 0
            fprintf('  無法對應隊名：%s @ %s（請在 team_index 的 KEY 加關鍵字）\n', odds.away{i}, odds.home{i});
            continue;
        end
        [pr, W, L] = ratings_before(gameDay(i), gDay, ai, hi, awayWon, n, cal.alpha);
        if min(W(ta) + L(ta), W(th) + L(th)) < MIN_GAMES
            fprintf('  %s @ %s：有隊伍本季不到 %d 場，模型不可靠，略過\n', odds.away{i}, odds.home{i}, MIN_GAMES);
            continue;
        end
        pH   = tsl_calibrate(pr(th) / (pr(th) + pr(ta)), cal.a, cal.b);
        oA   = odds.awayOdds(i);  oH = odds.homeOdds(i);
        over = 1 / oA + 1 / oH;                          % 兩邊隱含機率加總（> 1 的部分就是抽水）
        game = sprintf('%s @ %s', odds.away{i}, odds.home{i});
        st   = start_label(odds.start{i});
        R(end + 1) = make_row(st, game, '客', odds.away{i}, W(ta), L(ta), 1 - pH, oA, (1 / oA) / over, over); %#ok<AGROW>
        R(end + 1) = make_row(st, game, '主', odds.home{i}, W(th), L(th), pH,     oH, (1 / oH) / over, over); %#ok<AGROW>
    end
    if isempty(R), error('沒有可預測的比賽。'); end

    % ------------------------------------------------------------------
    %  4. 輸出（依 EV 由高到低）
    % ------------------------------------------------------------------
    [~, ord] = sort([R.ev], 'descend');
    R = R(ord);
    fprintf('\n模型：依勝率跳轉 PageRank α=%g，校準 a=%.3f b=%+.3f；運彩平均抽水 %.1f%%\n', ...
        cal.alpha, cal.a, cal.b, 100 * (mean([R.over]) - 1));
    fprintf('%-12s %-18s %-2s %-8s %7s %8s %8s %6s %8s\n', ...
        '開賽(台灣)', '比賽', '押', '隊伍', '戰績', '模型勝率', '運彩隱含', '賠率', 'EV');
    for r = R
        mark = '';
        if r.ev > minEV, mark = '  ◀ 正期望值'; end
        fprintf('%-12s %-18s %-2s %-8s %7s %7.1f%% %7.1f%% %6.2f %+7.1f%%%s\n', r.start, r.game, ...
            r.side, r.team, r.record, 100 * r.p, 100 * r.implied, r.odds, 100 * r.ev, mark);
    end

    [~, base] = fileparts(oddsFile);
    out = fullfile(here, ['tsl_predictions_' strrep(base, 'tsl_odds_', '') '.csv']);
    fid = fopen(out, 'w', 'n', 'UTF-8');
    if fid < 0, error('無法寫入 %s', out); end
    fprintf(fid, '%s', char(65279));                     % BOM，讓 Excel 正確顯示中文
    fprintf(fid, 'start,game,side,team,record,model_prob,implied_prob,odds,ev\n');
    for r = R
        fprintf(fid, '%s,%s,%s,%s,%s,%.4f,%.4f,%.3f,%.4f\n', r.start, r.game, r.side, ...
            r.team, r.record, r.p, r.implied, r.odds, r.ev);
    end
    fclose(fid);
    fprintf('\n已存檔：%s\n', out);
    fprintf('提醒：模型不考慮先發投手與傷兵，EV 只是估計；運彩抽水約 15–17%%，長期要贏需要比市場更準的資訊。\n');
end

% ======================================================================
%  區域函式
% ======================================================================
function f = find_odds_file(here)
% 在專案資料夾與 ~/Downloads 找最新的 tsl_odds_*.csv
    dirs = {here, fullfile(getenv('HOME'), 'Downloads')};
    f = '';  best = -inf;
    for k = 1:numel(dirs)
        L = dir(fullfile(dirs{k}, 'tsl_odds_*.csv'));
        for j = 1:numel(L)
            if L(j).datenum > best
                best = L(j).datenum;  f = fullfile(dirs{k}, L(j).name);
            end
        end
    end
end

function odds = read_odds(f)
% 讀 tsl_odds_snippet.js 產生的 CSV（UTF-8、可能有 BOM、欄位有雙引號）
%   欄位：event_id, start, away, home, away_odds, home_odds
    fid = fopen(f, 'r', 'n', 'UTF-8');
    if fid < 0, error('無法開啟 %s', f); end
    fgetl(fid);                                          % 跳過標題列
    odds = struct('start', {{}}, 'away', {{}}, 'home', {{}}, 'awayOdds', [], 'homeOdds', []);
    while true
        line = fgetl(fid);
        if ~ischar(line), break; end
        line = strrep(strrep(line, char(65279), ''), '"', '');
        if isempty(strtrim(line)), continue; end
        c = strsplit(line, ',');
        odds.start{end + 1, 1} = strtrim(c{2});
        odds.away{end + 1, 1}  = strtrim(c{3});
        odds.home{end + 1, 1}  = strtrim(c{4});
        odds.awayOdds(end + 1, 1) = str2double(c{5});
        odds.homeOdds(end + 1, 1) = str2double(c{6});
    end
    fclose(fid);
end

function cal = read_calibration(f)
% 讀 tsl_calibration.csv（由 tsl_calibration_backtest 產生）
    if exist(f, 'file') ~= 2
        error('找不到 %s，請先執行 tsl_calibration_backtest。', f);
    end
    fid = fopen(f, 'r');
    fgetl(fid);
    c = strsplit(fgetl(fid), ',');
    fclose(fid);
    cal.alpha = str2double(c{1});  cal.a = str2double(c{2});  cal.b = str2double(c{3});
end

function d = us_date(ts)
% 運彩開賽時間（例如 2026-09-28T03:10:00.000+08:00）→ 美東日期的 datenum
%   MLB 球季期間美東為 UTC−4；比賽都在美東白天到晚上開打，用 UTC−4 換日期不會出錯
    local = datenum(ts(1:19), 'yyyy-mm-ddTHH:MM:SS');
    tz    = ts(end - 5:end);                             % 例如 +08:00
    off   = (str2double(tz(2:3)) + str2double(tz(5:6)) / 60) * (1 - 2 * (tz(1) == '-'));
    d     = floor(local - off / 24 - 4 / 24);
end

function s = start_label(ts)
% 開賽時間顯示成「月/日 時:分」（台灣時間）
    s = datestr(datenum(ts(1:19), 'yyyy-mm-ddTHH:MM:SS'), 'mm/dd HH:MM');
end

function k = team_index(name, ids, engName)
% 運彩中文隊名 → ids 中的位置（找不到回傳 0）
%   先用中文隊名關鍵字，再用英文隊名最後一個字（例如 Yankees）當備援
    KEY = {'天使', 108; '響尾蛇', 109; '金鶯', 110; '紅襪', 111; '小熊', 112; '紅人', 113;
           '守護者', 114; '洛磯', 115; '老虎', 116; '太空人', 117; '皇家', 118; '道奇', 119;
           '國民', 120; '大都會', 121; '運動家', 133; '海盜', 134; '教士', 135; '水手', 136;
           '巨人', 137; '紅雀', 138; '光芒', 139; '遊騎兵', 140; '藍鳥', 141; '雙城', 142;
           '費城人', 143; '勇士', 144; '白襪', 145; '馬林魚', 146; '洋基', 147; '釀酒人', 158};
    k = 0;
    for j = 1:size(KEY, 1)
        if ~isempty(strfind(name, KEY{j, 1}))
            k = find(ids == KEY{j, 2}, 1);
            if isempty(k), k = 0; end
            return;
        end
    end
    for j = 1:numel(engName)
        w = strsplit(engName{j}, ' ');
        if ~isempty(strfind(lower(name), lower(w{end})))
            k = j;  return;
        end
    end
end

function [pr, W, L] = ratings_before(day, gDay, ai, hi, awayWon, n, alpha)
% 只用 day 以前的比賽，建網絡並算依勝率跳轉的 PageRank
    AH = zeros(n);  AA = zeros(n);  W = zeros(n, 1);  L = zeros(n, 1);
    for t = find(gDay < day)'
        a = ai(t);  h = hi(t);
        if awayWon(t)
            AA(h, a) = AA(h, a) + 1;  W(a) = W(a) + 1;  L(h) = L(h) + 1;
        else
            AH(a, h) = AH(a, h) + 1;  W(h) = W(h) + 1;  L(a) = L(a) + 1;
        end
    end
    pr = tsl_pagerank(AH, AA, W, L, alpha);
end

function r = make_row(st, game, side, team, w, l, p, od, imp, over)
    r = struct('start', st, 'game', game, 'side', side, 'team', team, ...
               'record', sprintf('%d-%d', w, l), 'p', p, 'odds', od, ...
               'implied', imp, 'ev', p * od - 1, 'over', over);
end
