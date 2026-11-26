function tsl_calibration_backtest(firstYear, lastYear)
% TSL_CALIBRATION_BACKTEST  運彩「不讓分」用的勝率校準回測（預設 2016–2026）
%
%   下注要看期望值 EV = p × 賠率 − 1，所以 p 必須「準」，不只是猜對勝隊。
%   這裡用依勝率跳轉的 PageRank 逐日滾動預測（只用當天以前的比賽），
%   再加一層邏輯斯校準，修正主場優勢與過度自信：
%
%       pRaw = π_主 / (π_主 + π_客)
%       p_主 = σ( a · logit(pRaw) + b )
%
%   α、a、b 用「留一球季交叉驗證」挑選並評分；最後用全部球季重新擬合，
%   存成 tsl_calibration.csv 給 tsl_predict 使用。
%
%   執行：  tsl_calibration_backtest            % 2016–2026
%           tsl_calibration_backtest(2021, 2026)
%   需要 mlb_games_YYYY.csv 快取（沒有的話會自動從 MLB Stats API 下載）。

    if nargin < 2, firstYear = 2016; lastYear = 2026; end
    SEASONS   = firstYear:lastYear;
    MIN_GAMES = 10;                                   % 兩隊都打滿這麼多場才計分
    ALPHAS    = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.4];   % 要比較的阻尼係數
    here      = fileparts(mfilename('fullpath'));
    if isempty(here), here = pwd; end
    OUT       = fullfile(here, 'tsl_calibration.csv');

    % ------------------------------------------------------------------
    %  1. 每季逐日滾動，收集「各 α 的主隊原始勝率」與「主隊是否獲勝」
    % ------------------------------------------------------------------
    fprintf('逐日滾動回測中…\n');
    nS = numel(SEASONS);
    X = cell(nS, 1);  Y = cell(nS, 1);
    for s = 1:nS
        [X{s}, Y{s}] = tsl_rolling_probs(here, SEASONS(s), ALPHAS, MIN_GAMES);
        fprintf('  %d: %d 場\n', SEASONS(s), numel(Y{s}));
    end
    [~, k015] = min(abs(ALPHAS - 0.15));              % 未校準的對照組用 α = 0.15

    % ------------------------------------------------------------------
    %  2. 留一球季交叉驗證：用其他球季挑 α、擬合 a、b，在剩下那季評分
    % ------------------------------------------------------------------
    rawLL = 0;  calLL = 0;  nAll = 0;
    pCV = [];  yCV = [];
    picks = cell(nS, 1);
    for s = 1:nS
        tr  = setdiff(1:nS, s);
        [k, a, b] = best_fit(vertcat(X{tr}), vertcat(Y{tr}), ALPHAS);
        p   = tsl_calibrate(X{s}(:, k), a, b);
        rawLL = rawLL + tsl_logloss(X{s}(:, k015), Y{s}) * numel(Y{s});
        calLL = calLL + tsl_logloss(p, Y{s}) * numel(Y{s});
        nAll  = nAll + numel(Y{s});
        pCV = [pCV; p];  yCV = [yCV; Y{s}];          %#ok<AGROW>
        picks{s} = sprintf('%d: α=%g a=%.3f b=%+.3f', SEASONS(s), ALPHAS(k), a, b);
    end

    fprintf('\n%d–%d 共 %d 場（兩隊皆 ≥ %d 場），主隊實際勝率 %.4f\n', ...
        SEASONS(1), SEASONS(end), nAll, MIN_GAMES, mean(yCV));
    fprintf('  未校準（α=0.15，無主場）     樣本外 log loss %.5f\n', rawLL / nAll);
    fprintf('  校準後（α、a、b 留一季挑選） 樣本外 log loss %.5f\n', calLL / nAll);
    fprintf('  丟銅板                       log loss %.5f\n', log(2));
    fprintf('\n各季挑出的參數：\n');
    fprintf('  %s\n', picks{:});

    % ------------------------------------------------------------------
    %  3. 校準表：被看好那一方的預測勝率 vs 實際勝率
    % ------------------------------------------------------------------
    fav    = max(pCV, 1 - pCV);
    favWin = yCV;  favWin(pCV < 0.5) = 1 - yCV(pCV < 0.5);
    edges  = [0.5 0.525 0.55 0.575 0.6 0.625 0.65 1.0001];
    fprintf('\n校準表（樣本外，被看好方的預測勝率 vs 實際勝率）\n');
    fprintf('  %-13s %6s %8s %7s   %s\n', '預測區間', '場數', '平均預測', '實際', '不賠錢所需最低賠率');
    for i = 1:numel(edges) - 1
        m = fav >= edges(i) & fav < edges(i + 1);
        if ~any(m), continue; end
        fprintf('  %.3f–%.3f  %6d %8.3f %7.3f   %8.2f\n', edges(i), min(edges(i + 1), 1), ...
            sum(m), mean(fav(m)), mean(favWin(m)), 1 / mean(favWin(m)));
    end

    % ------------------------------------------------------------------
    %  4. 用全部球季重新擬合並存檔
    % ------------------------------------------------------------------
    [k, a, b] = best_fit(vertcat(X{:}), vertcat(Y{:}), ALPHAS);
    fid = fopen(OUT, 'w');
    if fid < 0, error('無法寫入 %s', OUT); end
    fprintf(fid, 'alpha,a,b,seasons,games,cv_logloss\n');
    fprintf(fid, '%g,%.6f,%.6f,%d-%d,%d,%.6f\n', ALPHAS(k), a, b, ...
        SEASONS(1), SEASONS(end), nAll, calLL / nAll);
    fclose(fid);
    fprintf('\n全部球季擬合：α=%g  a=%.4f  b=%+.4f（實力相當時主隊校準後勝率 %.3f）\n', ...
        ALPHAS(k), a, b, tsl_calibrate(0.5, a, b));
    fprintf('已存檔：%s\n', OUT);
end

% ======================================================================
%  區域函式
% ======================================================================
function [k, a, b] = best_fit(X, y, alphas)
% 對每個 α 擬合 a、b，挑校準後 log loss 最低的
    best = inf;
    for j = 1:numel(alphas)
        w  = tsl_fit_logistic(X(:, j), y);
        ll = tsl_logloss(tsl_calibrate(X(:, j), w(1), w(2)), y);
        if ll < best
            best = ll;  k = j;  a = w(1);  b = w(2);
        end
    end
end
