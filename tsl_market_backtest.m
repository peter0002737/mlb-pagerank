function tsl_market_backtest(firstYear, lastYear)
% TSL_MARKET_BACKTEST  模型 vs 國際盤：模型能不能贏博彩公司？（預設 2016–2026）
%
%   需要：mlb_games_YYYY.csv 與 espn_odds_YYYY.csv（由 tsl_fetch_espn_odds 下載）
%   執行：tsl_market_backtest  或  tsl_market_backtest(2021, 2026)
%
%   模型勝率：依勝率跳轉 PageRank 逐日滾動（只用當天以前的比賽），
%             校準參數 a、b 用「留一球季」擬合（評分那季的資料不參與擬合），α 讀自 tsl_calibration.csv
%   市場勝率：各家博彩公司去掉抽水後主隊勝率的中位數（espn_odds 的 p_home_mkt）
%
%   回答三個問題
%     1. 誰比較準？        比較模型與市場的 log loss
%     2. 模型有沒有新資訊？ 邏輯斯迴歸 y ~ σ(w模型·logit(p模型) + w市場·logit(p市場) + c)
%                          w模型 ≈ 0 代表市場已經知道模型知道的一切
%     3. 照模型下注賺不賺？ 每場挑 EV = p模型 × 賠率 − 1 較大的一邊，EV 超過門檻就押 1 單位，
%                          分別用「各家中位數賠率」與「各家最好賠率」計算報酬率（ROI）與 95% 信賴區間

    if nargin < 2, firstYear = 2016; lastYear = 2026; end
    here = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    MIN_GAMES  = 10;
    THRESHOLDS = [0 0.02 0.05 0.10];
    cal   = read_calibration(fullfile(here, 'tsl_calibration.csv'));
    ALL   = 2016:2026;                                     % 校準 a、b 用的訓練球季
    TEST  = firstYear:lastYear;

    % ------------------------------------------------------------------
    %  1. 模型：每季逐日滾動的原始勝率
    % ------------------------------------------------------------------
    fprintf('逐日滾動計算模型勝率（α = %g）…\n', cal.alpha);
    R = cell(numel(ALL), 1);  Y = cell(numel(ALL), 1);  I = cell(numel(ALL), 1);
    for s = 1:numel(ALL)
        [R{s}, Y{s}, I{s}] = tsl_rolling_probs(here, ALL(s), cal.alpha, MIN_GAMES);
    end

    % ------------------------------------------------------------------
    %  2. 每個測試球季：留一季校準模型、對上賠率
    % ------------------------------------------------------------------
    pM = [];  pK = [];  y = [];  season = [];
    oA = [];  oH = [];  bA = [];  bH = [];  nA = [];  nH = [];  over = [];
    for s = find(ismember(ALL, TEST))
        f = fullfile(here, sprintf('espn_odds_%d.csv', ALL(s)));
        if exist(f, 'file') ~= 2
            fprintf('  %d：沒有 %s，略過（請先執行 tsl_fetch_espn_odds(%d)）\n', ALL(s), f, ALL(s));
            continue;
        end
        tr = setdiff(1:numel(ALL), s);
        w  = tsl_fit_logistic(vertcat(R{tr}), vertcat(Y{tr}));
        p  = tsl_calibrate(R{s}, w(1), w(2));

        od  = read_odds(f);
        idx = match_odds(I{s}, od);                        % 每場模型預測對應到哪一列賠率（0 = 沒有）
        ok  = idx > 0;
        ok(ok) = od.n(idx(ok)) > 0;
        j  = idx(ok);
        fprintf('  %d：模型 %d 場，對上賠率 %d 場（%.0f%%），平均 %.1f 家博彩公司\n', ALL(s), ...
            numel(p), sum(ok), 100 * mean(ok), mean(od.n(j)));
        pM = [pM; p(ok)];               pK = [pK; od.pHome(j)];       %#ok<AGROW>
        y  = [y; Y{s}(ok)];             season = [season; repmat(ALL(s), sum(ok), 1)]; %#ok<AGROW>
        oA = [oA; od.aMed(j)];          oH = [oH; od.hMed(j)];        %#ok<AGROW>
        bA = [bA; od.aBest(j)];         bH = [bH; od.hBest(j)];       %#ok<AGROW>
        nA = [nA; od.pinA(j)];          nH = [nH; od.pinH(j)];        %#ok<AGROW>
        over = [over; od.over(j)];                                    %#ok<AGROW>
    end
    if isempty(y), error('沒有任何球季同時有模型與賠率資料。'); end
    N = numel(y);

    % ------------------------------------------------------------------
    %  3. 誰比較準？
    % ------------------------------------------------------------------
    fprintf('\n=== 1. 誰比較準？（%d 場，主隊實際勝率 %.3f，市場平均抽水 %.1f%%）===\n', ...
        N, mean(y), 100 * mean(over));
    fprintf('  %-22s log loss %.5f\n', '模型（PageRank＋校準）', tsl_logloss(pM, y));
    fprintf('  %-22s log loss %.5f\n', '市場（去水後中位數）', tsl_logloss(pK, y));
    fprintf('  %-22s log loss %.5f\n', '丟銅板', log(2));
    fprintf('  模型與市場勝率的相關係數 %.3f，平均差距 %.1f 個百分點\n', ...
        corr_(pM, pK), 100 * mean(abs(pM - pK)));

    % ------------------------------------------------------------------
    %  4. 模型在市場之外有沒有新資訊？
    % ------------------------------------------------------------------
    [w, se] = fit_with_se([pM pK], y);
    fprintf('\n=== 2. 模型有沒有市場不知道的資訊？===\n');
    fprintf('  y ~ σ(w模型·logit(p模型) + w市場·logit(p市場) + c)\n');
    fprintf('  w模型 = %+.3f（標準誤 %.3f，z = %+.1f）\n', w(1), se(1), w(1) / se(1));
    fprintf('  w市場 = %+.3f（標準誤 %.3f，z = %+.1f）\n', w(2), se(2), w(2) / se(2));
    fprintf('  |z| < 2 代表統計上看不出差別；w模型 顯著 > 0 才代表模型有市場沒有的資訊。\n');
    llC = 0;                                               % 留一季：合併模型在樣本外有沒有比市場好
    for s = unique(season)'
        te = season == s;
        wc = tsl_fit_logistic([pM(~te) pK(~te)], y(~te));
        F  = [log(pM(te) ./ (1 - pM(te))), log(pK(te) ./ (1 - pK(te))), ones(sum(te), 1)];
        llC = llC + tsl_logloss(1 ./ (1 + exp(-F * wc)), y(te)) * sum(te);
    end
    fprintf('  合併模型（留一季）樣本外 log loss %.5f，市場 %.5f\n', llC / N, tsl_logloss(pK, y));

    % ------------------------------------------------------------------
    %  5. 照模型下注
    % ------------------------------------------------------------------
    fprintf('\n=== 3. 照模型下注（每注 1 單位，挑 EV 較大的一邊）===\n');
    fprintf('  %-14s %6s %7s %7s %9s %18s\n', '賠率／門檻', '注數', '押中率', '平均賠率', 'ROI', '95% 信賴區間');
    names = {'各家中位數', '各家最好'};
    odds  = {[oA oH], [bA bH]};
    for o = 1:2
        for thr = THRESHOLDS
            [n, hit, avgO, roi, ci] = simulate(pM, y, odds{o}(:, 1), odds{o}(:, 2), thr);
            fprintf('  %-8s EV>%-4.0f%% %6d %6.1f%% %7.2f %+8.2f%%   [%+6.2f%%, %+6.2f%%]\n', ...
                names{o}, 100 * thr, n, 100 * hit, avgO, 100 * roi, 100 * ci(1), 100 * ci(2));
        end
    end
    hasP = ~isnan(nA) & ~isnan(nH);
    if any(hasP)
        [n, hit, avgO, roi, ci] = simulate(pM(hasP), y(hasP), nA(hasP), nH(hasP), 0);
        fprintf('  %-8s EV>0    %6d %6.1f%% %7.2f %+8.2f%%   [%+6.2f%%, %+6.2f%%]（只有 %d 場有 Pinnacle）\n', ...
            'Pinnacle', n, 100 * hit, avgO, 100 * roi, 100 * ci(1), 100 * ci(2), sum(hasP));
    end
    [n, hit, avgO, roi, ci] = simulate(ones(N, 1), y, oA, oH, -inf);   % 對照：p主=1 → 每場都押主隊
    fprintf('  %-14s %6d %6.1f%% %7.2f %+8.2f%%   [%+6.2f%%, %+6.2f%%]\n', ...
        '對照：全押主隊', n, 100 * hit, avgO, 100 * roi, 100 * ci(1), 100 * ci(2));

    % ------------------------------------------------------------------
    %  6. 各季結果（中位數賠率、EV > 0），並存檔
    % ------------------------------------------------------------------
    out = fullfile(here, 'tsl_market_backtest.csv');
    fid = fopen(out, 'w');
    fprintf(fid, 'season,games,model_logloss,market_logloss,bets,hit_rate,roi\n');
    fprintf('\n各季（中位數賠率、EV > 0）\n');
    fprintf('  %-6s %6s %10s %10s %6s %8s\n', '球季', '場數', '模型LL', '市場LL', '注數', 'ROI');
    for s = unique(season)'
        m = season == s;
        [n, hit, ~, roi] = simulate(pM(m), y(m), oA(m), oH(m), 0);
        lm = tsl_logloss(pM(m), y(m));  lk = tsl_logloss(pK(m), y(m));
        fprintf('  %-6d %6d %10.5f %10.5f %6d %+7.2f%%\n', s, sum(m), lm, lk, n, 100 * roi);
        fprintf(fid, '%d,%d,%.6f,%.6f,%d,%.5f,%.5f\n', s, sum(m), lm, lk, n, hit, roi);
    end
    fclose(fid);
    fprintf('已存檔：%s\n', out);
end

% ======================================================================
%  區域函式
% ======================================================================
function [n, hit, avgO, roi, ci] = simulate(pHome, y, oAway, oHome, thr)
% 每場挑 EV 較大的一邊，EV > thr 就押 1 單位；回傳注數、押中率、平均賠率、ROI、ROI 的 95% 信賴區間
    evH = pHome .* oHome - 1;
    evA = (1 - pHome) .* oAway - 1;
    home = evH >= evA;
    ev   = max(evH, evA);
    bet  = ev > thr & ~isnan(ev);
    od   = oAway;  od(home) = oHome(home);
    win  = (home & y == 1) | (~home & y == 0);
    pnl  = win .* (od - 1) - ~win;                          % 贏：賺 賠率−1；輸：−1
    pnl  = pnl(bet);
    n    = numel(pnl);
    if n == 0, hit = NaN; avgO = NaN; roi = NaN; ci = [NaN NaN]; return; end
    hit  = mean(win(bet));
    avgO = mean(od(bet));
    roi  = mean(pnl);
    h    = 1.96 * std(pnl) / sqrt(n);
    ci   = [roi - h, roi + h];
end

function [w, se] = fit_with_se(P, y)
% 邏輯斯迴歸係數與標準誤（標準誤 = 資訊矩陣反矩陣對角線的平方根）
    w = tsl_fit_logistic(P, y);
    P = min(max(P, 1e-6), 1 - 1e-6);
    F = [log(P ./ (1 - P)), ones(size(P, 1), 1)];
    p = 1 ./ (1 + exp(-F * w));
    H = F' * (F .* repmat(p .* (1 - p), 1, size(F, 2)));
    se = sqrt(diag(inv(H)));
end

function idx = match_odds(info, od)
% 用「日期｜客隊｜主隊｜客隊得分｜主隊得分」對應；同一天找不到時再試前後一天（改期、時區）
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
    C = textscan(fid, '%s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %s', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    od.date  = C{2};   od.awayId = C{3};  od.homeId = C{4};
    od.awayScore = C{5};  od.homeScore = C{6};  od.n = C{7};
    od.aMed  = C{8};   od.hMed  = C{9};   od.aBest = C{10};  od.hBest = C{11};
    od.pHome = C{12};  od.over  = C{13};  od.pinA  = C{14};  od.pinH  = C{15};
end

function cal = read_calibration(f)
    fid = fopen(f, 'r');
    if fid < 0, error('找不到 %s，請先執行 tsl_calibration_backtest。', f); end
    fgetl(fid);
    c = strsplit(fgetl(fid), ',');
    fclose(fid);
    cal.alpha = str2double(c{1});  cal.a = str2double(c{2});  cal.b = str2double(c{3});
end

function r = corr_(x, z)
% 皮爾森相關係數（不依賴 Statistics Toolbox）
    x = x - mean(x);  z = z - mean(z);
    r = sum(x .* z) / sqrt(sum(x .^ 2) * sum(z .^ 2));
end
