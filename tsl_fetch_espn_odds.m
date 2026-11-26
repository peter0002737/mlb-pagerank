function tsl_fetch_espn_odds(season)
% TSL_FETCH_ESPN_ODDS  從 ESPN 公開 API 下載某一季 MLB 例行賽的國際盤獨贏賠率
%
%   tsl_fetch_espn_odds(2025)      → 寫成 espn_odds_2025.csv
%
%   資料來源（非官方文件化的 API，格式可能改變；僅供個人研究）
%     賽程： https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD
%     賠率： https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/ID/competitions/ID/odds
%
%   每場比賽會有數家博彩公司（2016 年有 Pinnacle、5Dimes、Bovada…；近年多為 DraftKings、ESPN BET），
%   這裡把每家的美式賠率換成十進位，再整理成一列：
%     n_books              有效博彩公司數
%     away_med, home_med   各家賠率的中位數（一般投注者拿得到的價格）
%     away_best, home_best 各家中最好的賠率（貨比三家的價格）
%     p_home_mkt           去掉抽水後的市場主隊勝率（各家中位數）
%     overround            兩邊隱含機率加總 − 1（抽水）
%     pinn_away, pinn_home Pinnacle 的賠率（沒有就是 NaN）
%   排除：場中賠率（Live）、consensus、numberfire、teamrankings（不是真正可下注的盤口），
%   以及和其他家差太多（去水勝率差 > 0.10，通常是主客隊標反）的資料。
%
%   比賽日期取自 mlb_games_YYYY.csv，對應時用「日期 + 兩隊 + 比分」，雙重賽也不會配錯。
%   可中斷：重新執行時會跳過檔案裡已經抓過的日期。每場一次請求，一季約 2,400 次，需時數十分鐘。

    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    gfile = fullfile(here, sprintf('mlb_games_%d.csv', season));
    if exist(gfile, 'file') ~= 2, tsl_fetch_games(season, gfile); end
    g     = tsl_read_games(gfile);
    days  = unique(g.date);
    out   = fullfile(here, sprintf('espn_odds_%d.csv', season));
    HEAD  = ['event_id,date,away_id,home_id,away_score,home_score,n_books,away_med,home_med,' ...
             'away_best,home_best,p_home_mkt,overround,pinn_away,pinn_home,books'];

    % --- 已抓過的日期（可中斷續抓）---
    done = {};
    if exist(out, 'file') == 2
        fid = fopen(out, 'r');
        C = textscan(fid, '%s %s %*[^\n]', 'Delimiter', ',', 'HeaderLines', 1);
        fclose(fid);
        done = unique(C{2});
    else
        fid = fopen(out, 'w');  fprintf(fid, '%s\n', HEAD);  fclose(fid);
    end
    todo = setdiff(days, done);
    fprintf('%d：共 %d 個比賽日，已抓 %d 天，剩 %d 天\n', season, numel(days), ...
        numel(days) - numel(todo), numel(todo));

    opts = weboptions('Timeout', 60, 'ContentType', 'text');
    nGame = 0;  nOdds = 0;  t0 = tic;
    for k = 1:numel(todo)
        day = todo{k};
        sb  = get_json(sprintf(['https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/' ...
                                'scoreboard?dates=%s'], strrep(day, '-', '')), opts);
        rows = {};
        for e = 1:numel(sb.events)
            ev = elem(sb.events, e);
            if ev.season.type ~= 2 || ~strcmp(ev.status.type.name, 'STATUS_FINAL'), continue; end
            c  = elem(ev.competitions, 1);
            [aId, hId, aSc, hSc] = teams(c);
            if isnan(aId) || isnan(hId), continue; end
            eid = ev.id;
            if ~ischar(eid), eid = sprintf('%d', eid); end
            od = get_json(sprintf(['https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/' ...
                                   'events/%s/competitions/%s/odds'], eid, eid), opts);
            r  = summarize_odds(od);
            nGame = nGame + 1;  nOdds = nOdds + (r.n > 0);
            rows{end + 1} = sprintf('%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f,%s', ...
                eid, day, aId, hId, aSc, hSc, r.n, r.aMed, r.hMed, r.aBest, r.hBest, ...
                r.pHome, r.over, r.pinA, r.pinH, r.books); %#ok<AGROW>
            pause(0.05);                                   % 放慢一點，對 ESPN 客氣
        end
        if ~isempty(rows)
            fid = fopen(out, 'a');                         % 每天寫一次，可隨時中斷
            fprintf(fid, '%s\n', rows{:});
            fclose(fid);
        end
        if mod(k, 10) == 0 || k == numel(todo)
            fprintf('  %s 完成（%d/%d 天，%d 場，其中 %d 場有賠率，已花 %.0f 秒）\n', ...
                day, k, numel(todo), nGame, nOdds, toc(t0));
        end
    end
    fprintf('已存檔：%s\n', out);
end

% ======================================================================
%  區域函式
% ======================================================================
function s = get_json(url, opts)
% 讀 JSON，失敗時重試三次
    for attempt = 1:4
        try
            s = jsondecode(webread(url, opts));
            return;
        catch err
            if attempt == 4, rethrow(err); end
            pause(3 * attempt);
        end
    end
end

function x = elem(arr, i)
% jsondecode 可能回傳 struct 陣列或 cell 陣列；統一取出第 i 個元素
    if iscell(arr), x = arr{i}; else, x = arr(i); end
end

function [aId, hId, aSc, hSc] = teams(c)
% 從 ESPN 的 competitors 取出客、主隊（換成 MLB Stats API 的球隊 id）與比分
    aId = NaN;  hId = NaN;  aSc = NaN;  hSc = NaN;
    for j = 1:numel(c.competitors)
        t  = elem(c.competitors, j);
        id = espn_to_mlb(num(t.team.id));
        sc = num(t.score);
        if strcmp(t.homeAway, 'home'), hId = id; hSc = sc; else, aId = id; aSc = sc; end
    end
end

function v = num(x)
% ESPN 有時給字串（"14"）、有時給數字；統一轉成數字
    if ischar(x), v = str2double(x); else, v = double(x); end
end

function id = espn_to_mlb(e)
% ESPN 球隊 id → MLB Stats API 球隊 id（30 隊固定對照）
    T = [ 1 110;  2 111;  3 108;  4 145;  5 114;  6 116;  7 118;  8 158;  9 142; 10 147;
         11 133; 12 136; 13 140; 14 141; 15 144; 16 112; 17 113; 18 117; 19 119; 20 120;
         21 121; 22 143; 23 134; 24 138; 25 135; 26 137; 27 115; 28 146; 29 109; 30 139];
    k  = find(T(:, 1) == e, 1);
    if isempty(k), id = NaN; else, id = T(k, 2); end
end

function r = summarize_odds(od)
% 把各家博彩公司的獨贏賠率整理成一列（見檔頭說明）
    r = struct('n', 0, 'aMed', NaN, 'hMed', NaN, 'aBest', NaN, 'hBest', NaN, ...
               'pHome', NaN, 'over', NaN, 'pinA', NaN, 'pinH', NaN, 'books', '');
    if ~isfield(od, 'items') || isempty(od.items), return; end
    name = {};  dA = [];  dH = [];
    for j = 1:numel(od.items)
        it = elem(od.items, j);
        if ~isfield(it, 'provider') || ~isfield(it.provider, 'name'), continue; end
        nm = it.provider.name;
        if ~isempty(regexpi(nm, 'live|consensus|numberfire|teamrankings', 'once')), continue; end
        if ~isfield(it, 'awayTeamOdds') || ~isfield(it, 'homeTeamOdds'), continue; end
        mA = moneyline(it.awayTeamOdds);  mH = moneyline(it.homeTeamOdds);
        if isnan(mA) || isnan(mH) || abs(mA) < 100 || abs(mH) < 100, continue; end
        name{end + 1} = strrep(nm, ',', ' ');  %#ok<AGROW>
        dA(end + 1) = american_to_decimal(mA);  %#ok<AGROW>
        dH(end + 1) = american_to_decimal(mH);  %#ok<AGROW>
    end
    if isempty(dA), return; end
    p  = (1 ./ dH) ./ (1 ./ dA + 1 ./ dH);                 % 每家去水後的主隊勝率
    ok = abs(p - median(p)) <= 0.10;                       % 排除主客標反等異常資料
    name = name(ok);  dA = dA(ok);  dH = dH(ok);  p = p(ok);
    r.n     = numel(dA);
    r.aMed  = median(dA);   r.hMed  = median(dH);
    r.aBest = max(dA);      r.hBest = max(dH);
    r.pHome = median(p);
    r.over  = median(1 ./ dA + 1 ./ dH) - 1;
    pin = find(~cellfun(@isempty, regexpi(name, 'pinnacle', 'once')), 1);
    if ~isempty(pin), r.pinA = dA(pin);  r.pinH = dH(pin); end
    r.books = strjoin(name, ';');
end

function m = moneyline(side)
% 取美式獨贏賠率：優先用收盤 close.moneyLine.american（字串，如 "+189"），否則用 moneyLine（數字）
    m = NaN;
    if isfield(side, 'close') && isstruct(side.close) && isfield(side.close, 'moneyLine') ...
            && isstruct(side.close.moneyLine) && isfield(side.close.moneyLine, 'american')
        s = side.close.moneyLine.american;
        if strcmpi(s, 'EVEN'), m = 100; else, m = str2double(strrep(s, '+', '')); end
    end
    if isnan(m) && isfield(side, 'moneyLine') && isnumeric(side.moneyLine) && ~isempty(side.moneyLine)
        m = double(side.moneyLine);
    end
end

function d = american_to_decimal(m)
% 美式賠率 → 十進位賠率（含本金）：+150 → 2.50，−180 → 1.556
    if m > 0, d = 1 + m / 100; else, d = 1 + 100 / abs(m); end
end
