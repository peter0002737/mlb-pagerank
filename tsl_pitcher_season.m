function tsl_pitcher_season(trainYear, testYear, halfLife)
% TSL_PITCHER_SEASON  整季檢驗「先發投手」與「近期比賽加權」：用前一季擬合、下一季測試
%
%   tsl_pitcher_season                 % 2025 擬合，2026 測試，近期加權半衰期 60 天
%   tsl_pitcher_season(2025, 2026, 30)
%
%   模型（主隊勝率 p = σ(特徵 × 係數)，係數全部只用 trainYear 擬合，testYear 完全沒看過）
%     基本        logit(pRaw)＋截距                         pRaw = 依勝率跳轉 PageRank 的主隊原始勝率
%     近期        同上，但 PageRank 用近期加權網絡
%     基本＋ERA   再加「客隊先發 ERA* − 主隊先發 ERA*」
%     基本＋FIP   再加「客隊先發 FIP* − 主隊先發 FIP*」
%     近期＋FIP   近期加權 PageRank＋先發 FIP 差
%   先發投手
%     名單：MLB Stats API 賽程的 probablePitcher（賽前公布的預定先發）
%     能力：該投手本季到前一天為止的成績，向聯盟平均收縮（IP0 = 30 局）
%           ERA* = 9 × (自責分 + 4.20 × IP0/9) / (局數 + IP0)
%           FIP* = (13·HR + 3·(BB+HBP) − 2·K + (4.20 − 3.10) × IP0) / (局數 + IP0) + 3.10
%           （FIP 只看全壘打、保送、觸身球、三振，比 ERA 少受守備與運氣影響）
%     查不到先發或本季還沒出賽的投手，就用聯盟平均 4.20。
%   評估（testYear）
%     log loss（和市場比）、模型對市場的增量資訊（邏輯斯迴歸中模型的係數）、
%     照模型下注（EV > 0 押 100 元）的報酬
%
%   資料快取：sp_probable_YYYY.csv（每場預定先發）、sp_logs_YYYY.csv（投手逐場成績），
%   第一次執行會下載（每季約 400 位投手，數分鐘），之後直接讀取。

    if nargin < 1 || isempty(trainYear), trainYear = 2025; end
    if nargin < 2 || isempty(testYear),  testYear  = 2026; end
    if nargin < 3 || isempty(halfLife),  halfLife  = 60;   end
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    MIN_GAMES = 10;  STAKE = 100;
    cal = read_calibration(fullfile(here, 'tsl_calibration.csv'));
    names = {'基本', '近期', '基本＋ERA', '基本＋FIP', '近期＋FIP'};

    % ------------------------------------------------------------------
    %  1. 兩季的特徵
    % ------------------------------------------------------------------
    D = cell(2, 1);  yrs = [trainYear testYear];
    for s = 1:2
        fprintf('準備 %d 的資料…\n', yrs(s));
        [xB, y, info] = tsl_rolling_probs(here, yrs(s), cal.alpha, MIN_GAMES);
        [xR, yR, infoR] = tsl_rolling_probs(here, yrs(s), cal.alpha, MIN_GAMES, halfLife);
        assert(isequal(yR, y) && isequal(infoR.gamePk, info.gamePk));
        [dE, dF, cover] = pitcher_diff(here, yrs(s), info);
        fprintf('  %d：%d 場，其中 %.1f%% 兩隊都有預定先發\n', yrs(s), numel(y), 100 * cover);
        lB = logit(xB);  lR = logit(xR);  one = ones(size(y));
        D{s}.F = {[lB one], [lR one], [lB dE one], [lB dF one], [lR dF one]};
        D{s}.y = y;  D{s}.info = info;  D{s}.dE = dE;  D{s}.dF = dF;
    end

    % ------------------------------------------------------------------
    %  2. 用 trainYear 擬合係數
    % ------------------------------------------------------------------
    fprintf('\n用 %d 擬合的係數（主隊 logit = …）\n', trainYear);
    Wt = cell(5, 1);
    for m = 1:5
        Wt{m} = fit_logit(D{1}.F{m}, D{1}.y);
        switch m
            case {1, 2}, fprintf('  %-10s %.3f·logit(pRaw) %+.3f\n', names{m}, Wt{m}(1), Wt{m}(2));
            otherwise,  fprintf('  %-10s %.3f·logit(pRaw) %+.3f·(客−主 先發%s) %+.3f\n', names{m}, ...
                            Wt{m}(1), Wt{m}(2), names{m}(end-2:end), Wt{m}(3));
        end
    end
    fprintf('  （理論近似：每 1 點 ERA 差約 0.254）\n');

    % ------------------------------------------------------------------
    %  3. testYear：對上市場賠率
    % ------------------------------------------------------------------
    T  = D{2};
    od = read_odds(fullfile(here, sprintf('espn_odds_%d.csv', testYear)));
    idx = match_odds(T.info, od);
    ok  = idx > 0;  ok(ok) = od.n(idx(ok)) > 0;
    j   = idx(ok);  y = T.y(ok);
    pK  = od.pHome(j);  oA = od.aMed(j);  oH = od.hMed(j);
    P   = zeros(sum(ok), 5);
    for m = 1:5
        F = T.F{m}(ok, :);
        P(:, m) = 1 ./ (1 + exp(-F * Wt{m}));
    end
    N = sum(ok);
    fprintf('\n%d 測試（%d 場有賠率，市場平均抽水 %.2f%%）\n', testYear, N, 100 * mean(od.over(j)));
    fprintf('  %-10s %9s %10s %12s %14s %18s\n', '模型', 'log loss', '與市場差距', '模型係數 w', '（z 值）', '下注 ROI（EV>0）');
    fprintf('  %-10s %9.5f\n', '市場', tsl_logloss(pK, y));
    for m = 1:5
        [w, se] = fit_with_se([logit(P(:, m)) logit(pK) ones(N, 1)], y);
        [nb, roi, ci] = simulate(P(:, m), y, oA, oH);
        fprintf('  %-10s %9.5f %9.1f 點 %+11.3f  (z = %+5.1f)   %+6.2f%% [%+.1f%%, %+.1f%%]，%d 注\n', ...
            names{m}, tsl_logloss(P(:, m), y), 100 * mean(abs(P(:, m) - pK)), w(1), w(1) / se(1), ...
            100 * roi, 100 * ci(1), 100 * ci(2), nb);
    end
    fprintf('  %-10s %9.5f\n', '丟銅板', log(2));
    fprintf(['\n  模型係數 w：把模型和市場的 logit 一起放進邏輯斯迴歸時，模型的係數。\n' ...
             '  w 顯著 > 0（z > 2）才代表模型有市場沒有的資訊。\n']);

    % --- 依月份看先發投手的貢獻 ---
    mon = cellfun(@(s) s(1:7), T.info.date(ok), 'UniformOutput', false);
    [um, ~, mi] = unique(mon);
    fprintf('\n各月 log loss\n  %-8s %6s %9s', '月份', '場數', '市場');
    fprintf(' %9s', names{:});  fprintf('\n');
    for k = 1:numel(um)
        r = mi == k;
        fprintf('  %-8s %6d %9.4f', um{k}, sum(r), tsl_logloss(pK(r), y(r)));
        for m = 1:5, fprintf(' %9.4f', tsl_logloss(P(r, m), y(r))); end
        fprintf('\n');
    end

    % --- 存檔 ---
    out = fullfile(here, sprintf('tsl_pitcher_season_%d.csv', testYear));
    fid = fopen(out, 'w');
    fprintf(fid, 'gamePk,date,home_win,p_market,p_base,p_recent,p_base_era,p_base_fip,p_recent_fip,d_era,d_fip\n');
    ii = find(ok);
    for k = 1:N
        i = ii(k);
        fprintf(fid, '%d,%s,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.4f\n', T.info.gamePk(i), ...
            T.info.date{i}, y(k), pK(k), P(k, :), T.dE(i), T.dF(i));
    end
    fclose(fid);
    fprintf('\n已存檔：%s\n', out);
end

% ======================================================================
%  區域函式：先發投手
% ======================================================================
function [dE, dF, cover] = pitcher_diff(here, season, info)
% 每場「客隊先發 − 主隊先發」的收縮 ERA 與 FIP 差（正值＝主隊先發比較好）
    LG = 4.20;  CF = 3.10;  IP0 = 30;
    pp = probables(here, season);                      % gamePk, awayPid, homePid
    lg = logs(here, season, unique([pp.away(pp.away > 0); pp.home(pp.home > 0)]));
    n  = numel(info.gamePk);
    dE = zeros(n, 1);  dF = zeros(n, 1);  both = false(n, 1);
    for i = 1:n
        k = find(pp.pk == info.gamePk(i), 1, 'last');    % 延賽的比賽會出現多次，取最後（實際比賽日）那筆
        d = datenum(info.date{i}, 'yyyy-mm-dd');
        if isempty(k), continue; end
        [eA, fA, okA] = rate(lg, pp.away(k), d);
        [eH, fH, okH] = rate(lg, pp.home(k), d);
        dE(i) = eA - eH;  dF(i) = fA - fH;  both(i) = okA && okH;
    end
    cover = mean(both);

    function [e, f, ok] = rate(lg, pid, d)
        e = LG;  f = LG;  ok = pid > 0;
        if ~ok, return; end
        r = lg.pid == pid & lg.date < d;
        ip = sum(lg.ip(r));
        e = 9 * (sum(lg.er(r)) + LG * IP0 / 9) / (ip + IP0);
        f = (13 * sum(lg.hr(r)) + 3 * (sum(lg.bb(r)) + sum(lg.hbp(r))) - 2 * sum(lg.so(r)) ...
             + (LG - CF) * IP0) / (ip + IP0) + CF;
    end
end

function pp = probables(here, season)
% 整季每場的預定先發（快取 sp_probable_YYYY.csv）
    f = fullfile(here, sprintf('sp_probable_%d.csv', season));
    if exist(f, 'file') ~= 2
        fprintf('  下載 %d 預定先發名單…\n', season);
        opts = weboptions('Timeout', 300, 'ContentType', 'text');
        sj = jsondecode(webread(sprintf(['https://statsapi.mlb.com/api/v1/schedule?sportId=1' ...
            '&gameType=R&startDate=%d-01-01&endDate=%d-12-31&hydrate=probablePitcher'], season, season), opts));
        fid = fopen(f, 'w');  fprintf(fid, 'gamePk,away_pid,home_pid\n');
        for a = 1:numel(sj.dates)
            dd = elem(sj.dates, a);
            for b = 1:numel(dd.games)
                gm = elem(dd.games, b);
                fprintf(fid, '%d,%d,%d\n', gm.gamePk, pid_of(gm.teams.away), pid_of(gm.teams.home));
            end
        end
        fclose(fid);
    end
    fid = fopen(f, 'r');
    C = textscan(fid, '%f %f %f', 'Delimiter', ',', 'HeaderLines', 1);
    fclose(fid);
    pp.pk = C{1};  pp.away = C{2};  pp.home = C{3};
end

function p = pid_of(tm)
    p = 0;
    if isfield(tm, 'probablePitcher') && ~isempty(tm.probablePitcher), p = tm.probablePitcher.id; end
end

function lg = logs(here, season, pids)
% 投手逐場成績（快取 sp_logs_YYYY.csv；可中斷續抓）
    f = fullfile(here, sprintf('sp_logs_%d.csv', season));
    done = [];
    if exist(f, 'file') == 2
        lg = read_logs(f);  done = unique(lg.pid);
    else
        fid = fopen(f, 'w');  fprintf(fid, 'pid,date,ip,er,so,bb,hbp,hr\n');  fclose(fid);
    end
    todo = setdiff(pids, done);
    if ~isempty(todo)
        fprintf('  下載 %d 位投手的 %d 逐場成績…\n', numel(todo), season);
        opts = weboptions('Timeout', 60, 'ContentType', 'text');
        for k = 1:numel(todo)
            js = get_json(sprintf(['https://statsapi.mlb.com/api/v1/people/%d/stats' ...
                '?stats=gameLog&season=%d&group=pitching'], todo(k), season), opts);
            rows = sprintf('%d,0,0,0,0,0,0,0\n', todo(k));     % 沒有任何紀錄也留一列，代表已抓過
            if ~isempty(js.stats)
                st = elem(js.stats, 1);
                if isfield(st, 'splits')
                    for i = 1:numel(st.splits)
                        x = elem(st.splits, i);  s = x.stat;
                        rows = [rows sprintf('%d,%d,%.4f,%d,%d,%d,%d,%d\n', todo(k), ...
                            datenum(x.date, 'yyyy-mm-dd'), innings(s.inningsPitched), s.earnedRuns, ...
                            s.strikeOuts, s.baseOnBalls, s.hitByPitch, s.homeRuns)]; %#ok<AGROW>
                    end
                end
            end
            fid = fopen(f, 'a');  fprintf(fid, '%s', rows);  fclose(fid);
            if mod(k, 50) == 0, fprintf('    %d / %d\n', k, numel(todo)); end
            pause(0.05);
        end
    end
    lg = read_logs(f);
end

function lg = read_logs(f)
    fid = fopen(f, 'r');
    C = textscan(fid, '%f %f %f %f %f %f %f %f', 'Delimiter', ',', 'HeaderLines', 1);
    fclose(fid);
    lg.pid = C{1};  lg.date = C{2};  lg.ip = C{3};  lg.er = C{4};
    lg.so = C{5};  lg.bb = C{6};  lg.hbp = C{7};  lg.hr = C{8};
end

function v = innings(s)
% "5.2" 代表 5 又 2/3 局
    if isnumeric(s), s = sprintf('%.1f', s); end
    p = sscanf(s, '%d.%d');
    v = p(1);  if numel(p) > 1, v = v + p(2) / 3; end
end

function s = get_json(url, opts)
    for attempt = 1:4
        try
            s = jsondecode(webread(url, opts));  return;
        catch err
            if attempt == 4, rethrow(err); end
            pause(3 * attempt);
        end
    end
end

% ======================================================================
%  區域函式：擬合與評估
% ======================================================================
function w = fit_logit(F, y)
% 一般特徵的邏輯斯迴歸（牛頓法）：y ~ σ(F·w)
    w = zeros(size(F, 2), 1);  w(1) = 1;
    for it = 1:50
        p = 1 ./ (1 + exp(-F * w));
        H = F' * (F .* repmat(p .* (1 - p), 1, size(F, 2))) + 1e-9 * eye(size(F, 2));
        step = H \ (F' * (y - p));
        w = w + step;
        if max(abs(step)) < 1e-10, break; end
    end
end

function [w, se] = fit_with_se(F, y)
    w = fit_logit(F, y);
    p = 1 ./ (1 + exp(-F * w));
    H = F' * (F .* repmat(p .* (1 - p), 1, size(F, 2)));
    se = sqrt(diag(inv(H)));
end

function [n, roi, ci] = simulate(pH, y, oA, oH)
% 每場挑 EV 較大的一邊，EV > 0 押 1 單位
    evH = pH .* oH - 1;  evA = (1 - pH) .* oA - 1;
    home = evH >= evA;  bet = max(evH, evA) > 0;
    od = oA;  od(home) = oH(home);
    win = (home & y == 1) | (~home & y == 0);
    r = win .* (od - 1) - ~win;  r = r(bet);
    n = numel(r);  roi = mean(r);  h = 1.96 * std(r) / sqrt(n);  ci = [roi - h, roi + h];
end

function z = logit(p)
    p = min(max(p, 1e-6), 1 - 1e-6);  z = log(p ./ (1 - p));
end

function idx = match_odds(info, od)
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
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s，請先執行 tsl_fetch_espn_odds。', f); end
    C = textscan(fid, '%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    od.date = C{2};  od.awayId = C{3};  od.homeId = C{4};
    od.awayScore = C{5};  od.homeScore = C{6};  od.n = C{7};
    od.aMed = C{8};  od.hMed = C{9};  od.pHome = C{12};  od.over = C{13};
end

function cal = read_calibration(f)
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s', f); end
    fgetl(fid);  c = strsplit(fgetl(fid), ',');  fclose(fid);
    cal.alpha = str2double(c{1});  cal.a = str2double(c{2});  cal.b = str2double(c{3});
end

function x = elem(arr, i)
    if iscell(arr), x = arr{i}; else, x = arr(i); end
end
