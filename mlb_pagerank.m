function mlb_pagerank()
% MLB_PAGERANK  以 PageRank 對 MLB 球隊做排名（單一檔案、不使用任何 Toolbox）
%
%   流程：抓取賽程 → CSV 快取 → 建「輸家 → 贏家」加權有向圖 → 阻尼矩陣 D
%         → Google 矩陣 G = D·P + (I-D)·E → 冪迭代求穩態分布 → 特徵向量驗證
%         → 輸出表格、相關係數與圖
%
%   不使用單一阻尼係數 α，而是每隊各自一個阻尼值 d_i，組成對角阻尼矩陣
%   D = diag(d_1, ..., d_n)，d_i 由該隊輸球場數決定（見 build_damping）。
%
%   執行方式：在 MATLAB 中切到本檔所在資料夾後輸入  mlb_pagerank

% ======================================================================
%  參數設定（要調整的東西都集中在這裡）
% ======================================================================
SEASON       = 2026;        % 球季年份
WEIGHT_MODE  = 'win';       % 邊權重模式：'win'     每勝 1
                            %             'margin'  分差
                            %             'log'     log(1 + 分差)
HOME_MODE    = 'data';      % 主客場加權：'none'  不分主客場
                            %             'data'  依全聯盟主場勝率 h 做反機率加權：
                            %                     主場勝 × 1/(2h)、客場勝 × 1/(2(1-h))
                            %             數值    自己指定 h（例如 0.54）
DAMP_SCALE   = 1;           % 阻尼矩陣的收縮常數 c：d_i = L_i / (L_i + c·mean(L))
                            %   c = 1 為預設；c 越小 d_i 越接近 1（越相信比賽結果）
DAMP_SWEEP   = [0.25 0.5 0.75 1 1.25 1.5 2 3 4];   % 敏感度分析的 c 範圍
TOL          = 1e-12;       % 冪迭代收斂門檻（L1 範數）
MAX_ITER     = 1000;        % 冪迭代最大次數
EXPECT_GAMES = 162;         % 每隊預期場數（僅用於檢查並提示，不影響計算）

outDir       = fileparts(mfilename('fullpath'));   % 輸出放在程式所在資料夾
if isempty(outDir), outDir = pwd; end
if ischar(HOME_MODE) && strcmpi(HOME_MODE, 'none')
    RUN_TAG = WEIGHT_MODE;                  % 檔名標籤，例如 win
else
    RUN_TAG = [WEIGHT_MODE '_home'];        % 有主客場加權時，例如 win_home
end
TAG_LABEL    = strrep(RUN_TAG, '_', '+');   % 圖標題用（避免底線被當成下標）
COMPARE_CSV  = fullfile(outDir, sprintf('mlb_pagerank_%d_compare.csv', SEASON));   % 各種加權方式的名次比較
GAMES_CSV    = fullfile(outDir, sprintf('mlb_games_%d.csv', SEASON));      % 比賽快取
RESULT_CSV   = fullfile(outDir, sprintf('mlb_pagerank_%d_%s.csv', SEASON, RUN_TAG));
FIG_SENS     = fullfile(outDir, sprintf('mlb_pagerank_%d_%s_damping.png', SEASON, RUN_TAG));
FIG_SCATTER  = fullfile(outDir, sprintf('mlb_pagerank_%d_%s_scatter.png', SEASON, RUN_TAG));
FIG_MATRIX   = fullfile(outDir, sprintf('mlb_pagerank_%d_%s_matrix.png',  SEASON, RUN_TAG));
FIG_MARKOV   = fullfile(outDir, sprintf('mlb_pagerank_%d_%s_markov.png',  SEASON, RUN_TAG));
MATRIX_CSV   = fullfile(outDir, sprintf('mlb_matrix_%%s_%d_%s.csv', SEASON, RUN_TAG));  % %s = A / P / D / G
POST_CSV     = fullfile(outDir, sprintf('mlb_postseason_%d.csv', SEASON));                        % 季後賽快取
SERIES_CSV   = fullfile(outDir, sprintf('mlb_playoff_series_%d_%s.csv',   SEASON, RUN_TAG));  % 系列賽預測 vs 實際
CHAMP_CSV    = fullfile(outDir, sprintf('mlb_playoff_champion_%d_%s.csv', SEASON, RUN_TAG));  % 奪冠機率
FIG_PLAYOFF  = fullfile(outDir, sprintf('mlb_pagerank_%d_%s_playoff.png', SEASON, RUN_TAG));
MARKOV_STEPS = 40;          % 馬可夫鏈收斂圖要畫的步數

% ======================================================================
%  1. 取得比賽資料（有快取就直接讀，沒有才向 API 抓）
% ======================================================================
if exist(GAMES_CSV, 'file') == 2
    fprintf('讀取快取：%s\n', GAMES_CSV);
else
    fprintf('快取不存在，從 MLB Stats API 抓取 %d 例行賽...\n', SEASON);
    fetch_games(SEASON, GAMES_CSV);
end
games = read_games_csv(GAMES_CSV);
fprintf('完賽場數（去重、去平手後）：%d\n', numel(games.gamePk));

% ======================================================================
%  2. 建立球隊索引與勝敗紀錄
% ======================================================================
[teamIds, firstIdx] = unique([games.awayId; games.homeId]);
allNames  = [games.awayName; games.homeName];
allAbbrs  = [games.awayAbbr; games.homeAbbr];
teamNames = allNames(firstIdx);
teamAbbrs = allAbbrs(firstIdx);
n         = numel(teamIds);

[~, awayIdx] = ismember(games.awayId, teamIds);   % 每場比賽的客隊索引 (1..n)
[~, homeIdx] = ismember(games.homeId, teamIds);   % 每場比賽的主隊索引 (1..n)

awayWon   = games.awayScore > games.homeScore;
winnerIdx = homeIdx;  winnerIdx(awayWon) = awayIdx(awayWon);
loserIdx  = awayIdx;  loserIdx(awayWon)  = homeIdx(awayWon);
margin    = abs(games.awayScore - games.homeScore);

wins   = accumarray(winnerIdx, 1, [n 1]);
losses = accumarray(loserIdx,  1, [n 1]);
played = wins + losses;
pct    = wins ./ played;

% 場數檢查
fprintf('球隊數：%d\n', n);
bad = find(played ~= EXPECT_GAMES);
if isempty(bad)
    fprintf('場數檢查：%d 隊皆為 %d 場 ✓\n', n, EXPECT_GAMES);
else
    fprintf('場數檢查：以下球隊不是 %d 場（球季未結束或有未補賽）：\n', EXPECT_GAMES);
    for k = bad'
        fprintf('    %-25s %d 場\n', teamNames{k}, played(k));
    end
end

% ======================================================================
%  3. 加權鄰接矩陣  A(i,j) = 「i 輸給 j」的累計權重（邊：輸家 → 贏家）
% ======================================================================
%  每場比賽的權重 = 基本權重（每勝 1 / 分差 / log(1+分差)）× 主客場係數。
%  主客場係數：主場贏球比較容易，所以主場勝打折、客場勝加成；
%  用反機率加權，係數經過正規化，全部比賽的平均係數仍為 1。
homePct = mean(~awayWon);                   % 全聯盟主場勝率 h
[w, wHome, wAway] = game_weights(margin, awayWon, WEIGHT_MODE, HOME_MODE, homePct);
fprintf('\n邊權重：模式 = %s；全聯盟主場勝率 h = %.4f；主場勝 × %.4f，客場勝 × %.4f\n', ...
    WEIGHT_MODE, homePct, wHome, wAway);
A = accumarray([loserIdx, winnerIdx], w, [n n]);

% ======================================================================
%  4. 轉移矩陣 P（列正規化；沒有出邊的列改為均勻分布）
% ======================================================================
P = build_transition(A);

% ======================================================================
%  4b. 阻尼矩陣 D（取代單一的 α）
% ======================================================================
%  d_i = 第 i 隊「沿著比賽結果走」的機率，1 - d_i = 隨機跳到任一隊的機率。
%  輸越多場的球隊，出邊（投出去的票）越多、證據越充分，就越相信它的 P 列；
%  輸很少場的強隊，出邊少、偶然性大，就多用隨機跳轉平滑。
d    = build_damping(losses, DAMP_SCALE);
Dmat = diag(d);

fprintf('\n阻尼矩陣 D = diag(d_i)，d_i = L_i / (L_i + %g × mean(L))，mean(L) = %.2f\n', ...
    DAMP_SCALE, mean(losses));
fprintf('    d_i 範圍 %.4f ~ %.4f，平均 %.4f\n', min(d), max(d), mean(d));

% ======================================================================
%  5. PageRank：冪迭代，並以特徵向量驗證
% ======================================================================
[pr, nIter, resid] = pagerank_power(P, d, TOL, MAX_ITER);
prEig   = pagerank_eig(P, d);
eigDiff = max(abs(pr - prEig));

fprintf('\n權重設定 = %s，阻尼矩陣 c = %g\n', TAG_LABEL, DAMP_SCALE);
fprintf('冪迭代：%d 次收斂，最後一步 L1 變化量 = %.2e，sum(pr) = %.12f\n', nIter, resid, sum(pr));
fprintf('特徵向量驗證：max|冪迭代 - eig| = %.2e', eigDiff);
if eigDiff < 1e-8
    fprintf('  → 一致 ✓\n');
else
    fprintf('  → 不一致 ✗（請檢查）\n');
end

% ======================================================================
%  5b. 馬可夫鏈性質檢查，並把 A、P、D、G 四個矩陣存成 CSV
% ======================================================================
%  把球隊當成狀態、G 當成一步轉移機率矩陣，就是一條 30 個狀態的馬可夫鏈：
%  「隨機球迷」目前支持第 i 隊時，下一步以機率 d_i 改支持「打敗第 i 隊的某一隊」
%  （依權重抽），以機率 1-d_i 隨機跳到任一隊；PageRank 就是穩態分布 π（πG = π）。
G        = google_matrix(P, d);
lam      = sort(abs(eig(G)), 'descend');             % 特徵值大小，由大到小
statRes  = max(abs(pr' * G - pr'));                  % πG = π 的殘差
irredP   = all(all((eye(n) + P) ^ (n - 1) > 0));     % P 本身是否不可約（強連通）

fprintf('\n馬可夫鏈檢查（狀態數 = %d）：\n', n);
fprintf('    P 列和與 1 的最大偏差 = %.2e，G 列和與 1 的最大偏差 = %.2e\n', ...
    max(abs(sum(P, 2) - 1)), max(abs(sum(G, 2) - 1)));
fprintf('    沒有出邊（全勝）的球隊數 = %d\n', nnz(sum(A, 2) == 0));
fprintf('    G 的最小元素 = %.5f > 0  → 不可約且非週期，穩態分布唯一\n', min(G(:)));
if irredP
    fprintf('    P 本身（不加跳轉）不可約：是\n');
else
    fprintf('    P 本身（不加跳轉）不可約：否\n');
end
fprintf('    G = D·P + (I-D)·E 與逐列公式的最大差 = %.2e\n', ...
    max(max(abs(G - (Dmat * P + (eye(n) - Dmat) * ones(n) / n)))));
fprintf('    最大特徵值 = %.12f，第二大 |λ2| = %.4f（理論上 ≤ max(d_i) = %.4f，決定收斂速度）\n', ...
    lam(1), lam(2), max(d));
fprintf('    穩態驗證 max|πG - π| = %.2e\n', statRes);

write_matrix_csv(sprintf(MATRIX_CSV, 'A'), A, teamAbbrs, '%g');
write_matrix_csv(sprintf(MATRIX_CSV, 'P'), P, teamAbbrs, '%.6f');
write_matrix_csv(sprintf(MATRIX_CSV, 'D'), Dmat, teamAbbrs, '%.6f');
write_matrix_csv(sprintf(MATRIX_CSV, 'G'), G, teamAbbrs, '%.6f');
fprintf('矩陣已存檔（列 = 來源/輸家，欄 = 目的/贏家）：%s\n', sprintf(MATRIX_CSV, 'A|P|D|G'));

% ======================================================================
%  6. 排名與 Spearman 等級相關
% ======================================================================
prRank  = competition_rank(-pr);     % PageRank 名次（1 = 最高）
pctRank = competition_rank(-pct);    % 勝率名次（1 = 最高，同勝率同名次）
rho     = spearman_corr(pr, pct);

[~, order] = sortrows([-pr, -pct]);  % 依 PageRank 由高到低列出

fprintf('\n%-4s %-24s %4s %4s %6s %7s %9s %7s %7s %5s\n', ...
    'PR#', 'Team', 'W', 'L', 'Pct', 'd_i', 'PageRank', 'PR名次', '勝率名次', '差');
fprintf('%s\n', repmat('-', 1, 90));
for k = order'
    fprintf('%-4d %-24s %4d %4d %6.3f %7.4f %9.5f %7d %8d %+6d\n', ...
        prRank(k), teamNames{k}, wins(k), losses(k), pct(k), d(k), pr(k), ...
        prRank(k), pctRank(k), pctRank(k) - prRank(k));
end
fprintf('\nPageRank 與勝率的 Spearman 等級相關 ρ = %.4f\n', rho);

% 結果存成 CSV
fid = fopen(RESULT_CSV, 'w');
if fid < 0, error('無法寫入 %s', RESULT_CSV); end
fprintf(fid, 'team,abbr,wins,losses,win_pct,damping_d,pagerank,pagerank_rank,win_pct_rank,rank_diff\n');
for k = order'
    fprintf(fid, '%s,%s,%d,%d,%.4f,%.6f,%.8f,%d,%d,%d\n', ...
        teamNames{k}, teamAbbrs{k}, wins(k), losses(k), pct(k), d(k), pr(k), ...
        prRank(k), pctRank(k), pctRank(k) - prRank(k));
end
fclose(fid);
fprintf('排名表已存檔：%s\n', RESULT_CSV);

% ======================================================================
%  6b. 加權方式比較：3 種基本權重 × 有無主客場加權，共 6 種設定
% ======================================================================
%  每種設定只換鄰接矩陣 A 的權重，阻尼矩陣 D 不變，各算一次 PageRank。
cmpW    = {'win', 'margin', 'log', 'win', 'margin', 'log'};
cmpH    = {'none', 'none', 'none', HOME_MODE, HOME_MODE, HOME_MODE};
if ischar(HOME_MODE) && strcmpi(HOME_MODE, 'none'), cmpH(4:6) = {'data'}; end
nC      = numel(cmpW);
cmpName = cell(1, nC);
prAll   = zeros(n, nC);
rkAll   = zeros(n, nC);
rhoAll  = zeros(1, nC);
for q = 1:nC
    wq = game_weights(margin, awayWon, cmpW{q}, cmpH{q}, homePct);
    Pq = build_transition(accumarray([loserIdx, winnerIdx], wq, [n n]));
    prAll(:, q) = pagerank_power(Pq, d, TOL, MAX_ITER);
    rkAll(:, q) = competition_rank(-prAll(:, q));
    rhoAll(q)   = spearman_corr(prAll(:, q), pct);
    if q <= 3, cmpName{q} = cmpW{q}; else, cmpName{q} = [cmpW{q} '+home']; end
end

fprintf('\n加權方式比較（阻尼矩陣相同，只換邊權重）：\n');
fprintf('%-13s %8s %10s   %s\n', 'Setting', 'rho', 'vs win', 'Top 5');
fprintf('%s\n', repmat('-', 1, 66));
for q = 1:nC
    [~, oq] = sort(prAll(:, q), 'descend');
    fprintf('%-13s %8.4f %10.4f   %s %s %s %s %s\n', cmpName{q}, rhoAll(q), ...
        spearman_corr(prAll(:, q), prAll(:, 1)), teamAbbrs{oq(1:5)});
end
fprintf('（rho = 與勝率的 Spearman 相關；vs win = 與「每勝 1、不分主客場」排名的 Spearman 相關）\n');

fid = fopen(COMPARE_CSV, 'w');
if fid < 0, error('無法寫入 %s', COMPARE_CSV); end
fprintf(fid, 'team,abbr,wins,losses,win_pct_rank');
fprintf(fid, ',rank_%s', cmpName{:});
fprintf(fid, ',pagerank_%s', cmpName{:});
fprintf(fid, '\n');
for k = order'
    fprintf(fid, '%s,%s,%d,%d,%d', teamNames{k}, teamAbbrs{k}, wins(k), losses(k), pctRank(k));
    fprintf(fid, ',%d', rkAll(k, :));
    fprintf(fid, ',%.8f', prAll(k, :));
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('加權方式比較已存檔：%s\n', COMPARE_CSV);

% ======================================================================
%  7. 阻尼矩陣敏感度分析：改變收縮常數 c，看排名是否穩定
% ======================================================================
nA       = numel(DAMP_SWEEP);
prSweep  = zeros(n, nA);    % 各 c 下的 PageRank
rkSweep  = zeros(n, nA);    % 各 c 下的名次
rhoSweep = zeros(1, nA);    % 各 c 下與勝率的 Spearman ρ
dMean    = zeros(1, nA);    % 各 c 下 d_i 的平均
for a = 1:nA
    dA            = build_damping(losses, DAMP_SWEEP(a));
    dMean(a)      = mean(dA);
    prSweep(:, a) = pagerank_power(P, dA, TOL, MAX_ITER);
    rkSweep(:, a) = competition_rank(-prSweep(:, a));
    rhoSweep(a)   = spearman_corr(prSweep(:, a), pct);
end
fprintf('\n阻尼矩陣敏感度（Spearman ρ 對勝率）：\n');
for a = 1:nA
    fprintf('    c = %-5g  平均 d = %.4f   ρ = %.4f\n', DAMP_SWEEP(a), dMean(a), rhoSweep(a));
end

colors = hsv(n) * 0.85;     % 每隊一個顏色（依預設 c 的名次排色）
cLo = DAMP_SWEEP(1);  cHi = DAMP_SWEEP(end);
fig1 = figure('Color', 'w', 'Position', [100 100 1400 700]);
try, theme(fig1, 'light'); catch, end   % 深色介面下仍輸出白底圖（舊版無 theme 則略過）

subplot(1, 2, 1); hold on;
for r = 1:n
    k = order(r);
    plot(DAMP_SWEEP, prSweep(k, :), '-o', 'Color', colors(r, :), ...
        'LineWidth', 1.1, 'MarkerSize', 3, 'MarkerFaceColor', colors(r, :));
end
plot([cLo cHi], [1 1] / n, 'k--');          % 均勻分布基準 1/n
set(gca, 'XScale', 'log', 'XTick', DAMP_SWEEP);
xlabel('c  in  d_i = L_i / (L_i + c \cdot mean(L))'); ylabel('PageRank');
title(sprintf('PageRank vs damping matrix  (%d, weight = %s)', SEASON, TAG_LABEL));
xlim([cLo cHi]); grid on; box on;

subplot(1, 2, 2); hold on;
for r = 1:n
    k = order(r);
    plot(DAMP_SWEEP, rkSweep(k, :), '-o', 'Color', colors(r, :), ...
        'LineWidth', 1.1, 'MarkerSize', 3, 'MarkerFaceColor', colors(r, :));
    text(cHi * 1.04, rkSweep(k, end), teamAbbrs{k}, ...
        'Color', colors(r, :), 'FontSize', 8);
    text(cLo / 1.04, rkSweep(k, 1), teamAbbrs{k}, ...
        'Color', colors(r, :), 'FontSize', 8, 'HorizontalAlignment', 'right');
end
set(gca, 'XScale', 'log', 'XTick', DAMP_SWEEP, 'YDir', 'reverse', 'YTick', 1:n, 'FontSize', 8);
xlabel('c  (larger c \rightarrow smaller d_i \rightarrow more random jumps)'); ylabel('PageRank rank');
title('Rank vs damping matrix');
xlim([cLo / 1.25, cHi * 1.25]); ylim([0.5, n + 0.5]);
grid on; box on;

print(fig1, '-dpng', '-r150', FIG_SENS);
fprintf('敏感度分析圖已存檔：%s\n', FIG_SENS);

% ======================================================================
%  8. PageRank 對勝率散佈圖
% ======================================================================
fig2 = figure('Color', 'w', 'Position', [150 150 900 700]);
try, theme(fig2, 'light'); catch, end   % 深色介面下仍輸出白底圖（舊版無 theme 則略過）
hold on;
% 最小平方迴歸線（僅作視覺參考）
c  = polyfit(pct, pr, 1);
xx = [min(pct) max(pct)];
plot(xx, polyval(c, xx), '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
plot(pct, pr, 'o', 'MarkerSize', 6, 'MarkerFaceColor', [0.15 0.45 0.8], ...
    'MarkerEdgeColor', 'w');
dx = (max(pct) - min(pct)) * 0.008;
for k = 1:n
    text(pct(k) + dx, pr(k), teamAbbrs{k}, 'FontSize', 8);
end
xlabel('Win percentage'); ylabel(sprintf('PageRank (damping matrix, c = %g)', DAMP_SCALE));
title(sprintf('MLB %d: PageRank vs Win%%  (weight = %s, Spearman \\rho = %.3f)', ...
    SEASON, TAG_LABEL, rho));
grid on; box on;

print(fig2, '-dpng', '-r150', FIG_SCATTER);
fprintf('散佈圖已存檔：%s\n', FIG_SCATTER);

% ======================================================================
%  9. 矩陣熱圖：鄰接矩陣 A 與轉移矩陣 P（球隊依 PageRank 由高到低排列）
% ======================================================================
lbl  = teamAbbrs(order);
fig3 = figure('Color', 'w', 'Position', [120 120 1500 720]);
try, theme(fig3, 'light'); catch, end   % 深色介面下仍輸出白底圖（舊版無 theme 則略過）

subplot(1, 2, 1);
imagesc(A(order, order)); axis square; colorbar;
set(gca, 'XTick', 1:n, 'XTickLabel', lbl, 'YTick', 1:n, 'YTickLabel', lbl, ...
    'FontSize', 7, 'XTickLabelRotation', 90);
xlabel('Winner (to)'); ylabel('Loser (from)');
title(sprintf('Adjacency matrix A  (weight = %s)', TAG_LABEL));

subplot(1, 2, 2);
imagesc(P(order, order)); axis square; colorbar;
set(gca, 'XTick', 1:n, 'XTickLabel', lbl, 'YTick', 1:n, 'YTickLabel', lbl, ...
    'FontSize', 7, 'XTickLabelRotation', 90);
xlabel('Next state (to)'); ylabel('Current state (from)');
title('Transition matrix P  (rows sum to 1)');

print(fig3, '-dpng', '-r150', FIG_MATRIX);
fprintf('矩陣熱圖已存檔：%s\n', FIG_MATRIX);

% ======================================================================
%  10. 馬可夫鏈：狀態轉移圖與收斂過程
% ======================================================================
fig4 = figure('Color', 'w', 'Position', [140 140 1500 720]);
try, theme(fig4, 'light'); catch, end   % 深色介面下仍輸出白底圖（舊版無 theme 則略過）

% --- (a) 狀態轉移圖：節點排在圓周上（依 PageRank 名次順時針），大小 ∝ PageRank ---
subplot(1, 2, 1); hold on; axis equal off;
ang = pi/2 - 2*pi*(0:n-1)'/n;               % 第 1 名在正上方
xy  = zeros(n, 2);
xy(order, :) = [cos(ang), sin(ang)];
% 所有轉移（淡灰色細線）
[ii, jj] = find(P > 0 & A > 0);
ex = [xy(ii, 1), xy(jj, 1), nan(numel(ii), 1)]';
ey = [xy(ii, 2), xy(jj, 2), nan(numel(ii), 1)]';
plot(ex(:), ey(:), '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.3);
% 每個狀態機率最大的那條轉移（藍線，三角形標在靠近目的地的一端）
for i = 1:n
    [pmax, j] = max(P(i, :));
    if sum(A(i, :)) == 0, continue; end     % 全勝隊的列是均勻分布，不畫
    p1 = xy(i, :);  p2 = xy(j, :);
    q  = p1 + 0.88 * (p2 - p1);
    plot([p1(1) p2(1)], [p1(2) p2(2)], '-', 'Color', [0.15 0.45 0.8], ...
        'LineWidth', 0.5 + 12 * pmax);
    plot(q(1), q(2), '^', 'MarkerSize', 5, 'MarkerFaceColor', [0.15 0.45 0.8], ...
        'MarkerEdgeColor', 'w');
end
% 節點與標籤
for k = 1:n
    plot(xy(k, 1), xy(k, 2), 'o', 'MarkerSize', 4 + 500 * (pr(k) - min(pr)), ...
        'MarkerFaceColor', [0.9 0.4 0.2], 'MarkerEdgeColor', 'w');
    text(1.13 * xy(k, 1), 1.13 * xy(k, 2), teamAbbrs{k}, 'FontSize', 8, ...
        'HorizontalAlignment', 'center');
end
xlim([-1.3 1.3]); ylim([-1.3 1.3]);
title({'Markov chain: states = teams, edges = loser \rightarrow winner', ...
       'blue = most likely next state of each team; node size \propto PageRank'});

% --- (b) 收斂過程：π(k) = π(0) G^k 與穩態 π 的 L1 距離 ---
subplot(1, 2, 2);
K     = MARKOV_STEPS;
worst = order(end);                         % PageRank 最低的球隊
v0    = [ones(1, n) / n; zeros(1, n)];      % 兩種起點：均勻分布、全部機率放在最後一名
v0(2, worst) = 1;
dist  = zeros(2, K + 1);
for s = 1:2
    v = v0(s, :);
    for k = 0:K
        dist(s, k + 1) = sum(abs(v - prEig'));   % 以特徵向量解當穩態參考
        v = v * G;
    end
end
dist = max(dist, 1e-17);                    % 避免 log(0)
semilogy(0:K, dist(1, :), '-o', 'LineWidth', 1.2, 'MarkerSize', 3); hold on;
semilogy(0:K, dist(2, :), '-s', 'LineWidth', 1.2, 'MarkerSize', 3);
semilogy(0:K, 2 * max(d) .^ (0:K), 'k--');
semilogy(0:K, dist(2, 1) * lam(2) .^ (0:K), 'k:');
grid on;
ylim([1e-16 10]); xlabel('Step k'); ylabel('|| \pi^{(k)} - \pi ||_1');
legend({'start: uniform', sprintf('start: all mass on %s', teamAbbrs{worst}), ...
    sprintf('bound 2 d_{max}^k (d_{max} = %.3f)', max(d)), ...
    sprintf('|\\lambda_2|^k (|\\lambda_2| = %.3f)', lam(2))}, 'Location', 'northeast');
title('Convergence of the Markov chain to its stationary distribution');

print(fig4, '-dpng', '-r150', FIG_MARKOV);
fprintf('馬可夫鏈圖已存檔：%s\n', FIG_MARKOV);

% ======================================================================
%  11. 季後賽回測：只用例行賽算出的強度，預測每個系列賽由誰晉級、幾比幾，
%      並計算各隊奪冠機率，再與實際結果比對
% ======================================================================
%  三個模型的差別只在「A 對 B 單場勝率」怎麼算：
%    PageRank：p = PR(A) / (PR(A) + PR(B))
%    勝率    ：log5 公式  p = a(1-b) / (a(1-b) + b(1-a))，a、b 為例行賽勝率
%    丟銅板  ：p = 0.5（對照組，只反映賽制與首輪輪空的影響）
%  系列賽視為各場獨立、勝率固定（不考慮主客場與投手輪值）。
if exist(POST_CSV, 'file') == 2
    fprintf('\n讀取季後賽快取：%s\n', POST_CSV);
else
    fprintf('\n季後賽快取不存在，從 MLB Stats API 抓取 %d 季後賽...\n', SEASON);
    fetch_postseason(SEASON, POST_CSV);
end
post = read_post_csv(POST_CSV);
if isempty(post.gamePk)
    fprintf('沒有已完賽的季後賽，略過季後賽回測。\n');
    return;
end

[~, pAway] = ismember(post.awayId, teamIds);
[~, pHome] = ismember(post.homeId, teamIds);
roundCodes = 'FDLW';                        % 外卡、分區系列賽、聯盟冠軍賽、世界大賽
rnd = zeros(size(pAway));
for i = 1:numel(rnd)
    rnd(i) = find(roundCodes == post.gameType{i}(1));
end
pWinner = pHome;
aw = post.awayScore > post.homeScore;
pWinner(aw) = pAway(aw);

% --- 把比賽歸成系列賽：同一輪、同兩隊 = 一個系列賽 ---
[ser, ~, sIdx] = unique([rnd, min(pAway, pHome), max(pAway, pHome)], 'rows');   % 依輪次排序
nS      = size(ser, 1);
sNeed   = zeros(nS, 1);      % 晉級所需勝場
sWinA   = zeros(nS, 1);      % A 隊（ser 第 2 欄）實際勝場
sWinB   = zeros(nS, 1);      % B 隊（ser 第 3 欄）實際勝場
sActual = zeros(nS, 1);      % 實際晉級隊（未打完為 0）
sName   = cell(nS, 1);
for s = 1:nS
    g = find(sIdx == s);
    sNeed(s) = (max(post.gamesInSeries(g)) + 1) / 2;
    sWinA(s) = sum(pWinner(g) == ser(s, 2));
    sWinB(s) = sum(pWinner(g) == ser(s, 3));
    sName{s} = post.series{g(1)};
    if sWinA(s) == sNeed(s), sActual(s) = ser(s, 2); end
    if sWinB(s) == sNeed(s), sActual(s) = ser(s, 3); end
end

% --- 三個模型的單場勝率矩陣 Q{m}(i,j) = i 對 j 的單場勝率 ---
modelNames = {'PageRank', 'WinPct', 'Coin'};
nM  = numel(modelNames);
PRm = repmat(pr, 1, n);
Wm  = repmat(pct, 1, n);
Q   = cell(nM, 1);
Q{1} = PRm ./ (PRm + PRm');
num  = Wm .* (1 - Wm');
Q{2} = num ./ (num + Wm' .* (1 - Wm));
Q{3} = 0.5 * ones(n);
for m = 1:nM
    Q{m}(1:n+1:end) = 0.5;                  % 對角線用不到，設 0.5 以免出現 NaN
end

% --- 逐系列賽預測 ---
predWin  = zeros(nS, nM);    % 預測晉級隊
predLose = zeros(nS, nM);    % 預測輸家的勝場數（例如 4-2 的 2）
pSerA    = zeros(nS, nM);    % A 隊贏下系列賽的機率
for s = 1:nS
    A2 = ser(s, 2);  B2 = ser(s, 3);
    for m = 1:nM
        [pSerA(s, m), kBest] = series_outcome(Q{m}(A2, B2), sNeed(s));
        if pSerA(s, m) >= 0.5, predWin(s, m) = A2; else, predWin(s, m) = B2; end
        predLose(s, m) = kBest;
    end
end

done     = sActual > 0;
actLose  = min(sWinA, sWinB);
hitWin   = predWin(:, 1:2) == repmat(sActual, 1, 2) & repmat(done, 1, 2);
hitScore = hitWin & predLose(:, 1:2) == repmat(actLose, 1, 2);

fprintf('\n季後賽系列賽回測（模型只用例行賽資料；阻尼矩陣 c = %g，權重 = %s）\n', DAMP_SCALE, TAG_LABEL);
fprintf('%-24s %-10s %7s %7s  %-10s %-10s %-10s %s\n', ...
    'Series', 'A vs B', 'P1(A)', 'PS(A)', 'PageRank', 'WinPct', 'Actual', 'PR / Pct');
fprintf('%s\n', repmat('-', 1, 98));
fid = fopen(SERIES_CSV, 'w');
if fid < 0, error('無法寫入 %s', SERIES_CSV); end
fprintf(fid, ['series,team_a,team_b,best_of,game_prob_a_pagerank,series_prob_a_pagerank,' ...
              'pred_pagerank,series_prob_a_winpct,pred_winpct,actual,' ...
              'pagerank_winner_hit,winpct_winner_hit,pagerank_score_hit,winpct_score_hit\n']);
marks = {'x', 'o'};
for s = 1:nS
    A2 = ser(s, 2);  B2 = ser(s, 3);
    str1 = sprintf('%s %d-%d', teamAbbrs{predWin(s, 1)}, sNeed(s), predLose(s, 1));
    str2 = sprintf('%s %d-%d', teamAbbrs{predWin(s, 2)}, sNeed(s), predLose(s, 2));
    if done(s)
        strA = sprintf('%s %d-%d', teamAbbrs{sActual(s)}, sNeed(s), actLose(s));
    else
        strA = sprintf('%d-%d (TBD)', sWinA(s), sWinB(s));
    end
    fprintf('%-24s %-10s %7.3f %7.3f  %-10s %-10s %-10s %s / %s\n', sName{s}, ...
        [teamAbbrs{A2} '-' teamAbbrs{B2}], Q{1}(A2, B2), pSerA(s, 1), str1, str2, strA, ...
        marks{hitWin(s, 1) + 1}, marks{hitWin(s, 2) + 1});
    fprintf(fid, '%s,%s,%s,%d,%.4f,%.4f,%s,%.4f,%s,%s,%d,%d,%d,%d\n', sName{s}, ...
        teamAbbrs{A2}, teamAbbrs{B2}, 2 * sNeed(s) - 1, Q{1}(A2, B2), pSerA(s, 1), str1, ...
        pSerA(s, 2), str2, strA, hitWin(s, 1), hitWin(s, 2), hitScore(s, 1), hitScore(s, 2));
end
fclose(fid);
fprintf('（P1(A) = A 隊單場勝率，PS(A) = A 隊贏下系列賽的機率，皆為 PageRank 模型；o = 猜中晉級隊）\n');
fprintf('晉級隊命中：PageRank %d / %d，勝率模型 %d / %d\n', ...
    sum(hitWin(:, 1)), sum(done), sum(hitWin(:, 2)), sum(done));
fprintf('場數比分也命中：PageRank %d / %d，勝率模型 %d / %d\n', ...
    sum(hitScore(:, 1)), sum(done), sum(hitScore(:, 2)), sum(done));
fprintf('系列賽預測已存檔：%s\n', SERIES_CSV);

% 各種加權設定的晉級隊命中數（PageRank 較高的一方視為預測晉級隊）
fprintf('\n各種加權設定的季後賽晉級隊命中數：\n');
for q = 1:nC
    pa = prAll(ser(:, 2), q);  pb = prAll(ser(:, 3), q);
    pw = ser(:, 3);  pw(pa >= pb) = ser(pa >= pb, 2);
    fprintf('    %-13s %d / %d\n', cmpName{q}, sum(pw == sActual & done), sum(done));
end

% --- 奪冠機率：依實際賽程樹，從第一輪往上精確計算（不是模擬）---
%  每個系列賽的每一邊，若該隊在更早的輪次出現過，這一邊就是「那個系列賽的勝者」，
%  否則是首輪輪空的種子。adv(:, s) = 各隊從系列賽 s 晉級的機率。
finalS = find(ser(:, 1) == 4, 1);
if isempty(finalS)
    fprintf('世界大賽尚未開打，略過奪冠機率。\n');
    return;
end
champ = zeros(n, nM);
for m = 1:nM
    adv = zeros(n, nS);
    for s = 1:nS
        d = zeros(n, 2);
        for side = 1:2
            t = ser(s, 1 + side);
            feeder = find(ser(1:s-1, 1) < ser(s, 1) & ...
                (ser(1:s-1, 2) == t | ser(1:s-1, 3) == t), 1, 'last');
            if isempty(feeder), d(t, side) = 1; else, d(:, side) = adv(:, feeder); end
        end
        Wn = series_win_matrix(Q{m}, sNeed(s));
        adv(:, s) = d(:, 1) .* (Wn * d(:, 2)) + d(:, 2) .* (Wn * d(:, 1));
    end
    champ(:, m) = adv(:, finalS);
end

inPost = find(any(champ > 0, 2));                    % 季後賽球隊
[~, o] = sort(champ(inPost, 1), 'descend');
inPost = inPost(o);
realChamp = sActual(finalS);

fprintf('\n奪冠機率回測（季後賽開打前、依實際對戰樹計算；機率總和 = %.6f）\n', sum(champ(:, 1)));
fprintf('%-24s %4s %4s %6s %10s %10s %10s\n', 'Team', 'W', 'L', 'PR#', 'PageRank', 'WinPct', 'Coin');
fprintf('%s\n', repmat('-', 1, 76));
fid = fopen(CHAMP_CSV, 'w');
if fid < 0, error('無法寫入 %s', CHAMP_CSV); end
fprintf(fid, 'team,abbr,wins,losses,pagerank_rank,p_champion_pagerank,p_champion_winpct,p_champion_coin,actual_champion\n');
for k = inPost'
    tag = '';
    if k == realChamp, tag = '  <== 實際冠軍'; end
    fprintf('%-24s %4d %4d %6d %9.1f%% %9.1f%% %9.1f%%%s\n', teamNames{k}, wins(k), losses(k), ...
        prRank(k), 100 * champ(k, 1), 100 * champ(k, 2), 100 * champ(k, 3), tag);
    fprintf(fid, '%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%d\n', teamNames{k}, teamAbbrs{k}, wins(k), ...
        losses(k), prRank(k), champ(k, 1), champ(k, 2), champ(k, 3), k == realChamp);
end
fclose(fid);
if realChamp > 0
    for m = 1:nM
        fprintf('實際冠軍 %s：%s 模型給的奪冠機率 %.1f%%，在 %d 支季後賽球隊中排第 %d\n', ...
            teamAbbrs{realChamp}, modelNames{m}, 100 * champ(realChamp, m), numel(inPost), ...
            1 + sum(champ(inPost, m) > champ(realChamp, m)));
    end
end
fprintf('奪冠機率已存檔：%s\n', CHAMP_CSV);

% --- 奪冠機率長條圖 ---
fig5 = figure('Color', 'w', 'Position', [160 160 1200 600]);
try, theme(fig5, 'light'); catch, end   % 深色介面下仍輸出白底圖（舊版無 theme 則略過）
bar(100 * champ(inPost, :));
set(gca, 'XTick', 1:numel(inPost), 'XTickLabel', teamAbbrs(inPost));
ylabel('Championship probability (%)'); grid on;
legend({'PageRank', 'Win% (log5)', 'Coin flip'}, 'Location', 'northeast');
if realChamp > 0
    title(sprintf('MLB %d postseason backtest: title odds before the playoffs  (actual champion: %s)', ...
        SEASON, teamAbbrs{realChamp}));
else
    title(sprintf('MLB %d postseason: title odds before the playoffs', SEASON));
end
print(fig5, '-dpng', '-r150', FIG_PLAYOFF);
fprintf('奪冠機率圖已存檔：%s\n', FIG_PLAYOFF);
end

% ######################################################################
%  以下為區域函式
% ######################################################################

function fetch_games(season, csvFile)
% 從 MLB Stats API 抓整季例行賽，只留已完賽 (codedGameState = 'F')、
% 依 gamePk 去重、去掉平手，寫成 CSV 快取。
    opts = weboptions('Timeout', 120, 'ContentType', 'text');

    % --- 球隊縮寫（畫圖標籤用）---
    urlTeams = sprintf('https://statsapi.mlb.com/api/v1/teams?sportId=1&season=%d', season);
    tj = jsondecode(webread(urlTeams, opts));
    nT = numel(tj.teams);
    abbrId = zeros(nT, 1);  abbrStr = cell(nT, 1);
    for i = 1:nT
        t = elem(tj.teams, i);
        abbrId(i)  = t.id;
        abbrStr{i} = t.abbreviation;
    end

    % --- 賽程 ---
    url = sprintf(['https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R' ...
                   '&startDate=%d-01-01&endDate=%d-12-31'], season, season);
    sj = jsondecode(webread(url, opts));

    % 先數總場數以便預先配置
    nMax = 0;
    for d = 1:numel(sj.dates)
        dd = elem(sj.dates, d);
        nMax = nMax + numel(dd.games);
    end
    pk = zeros(nMax, 1);  dt = cell(nMax, 1);
    aId = zeros(nMax, 1); hId = zeros(nMax, 1);
    aNm = cell(nMax, 1);  hNm = cell(nMax, 1);
    aSc = zeros(nMax, 1); hSc = zeros(nMax, 1);
    m = 0;

    for d = 1:numel(sj.dates)
        dd = elem(sj.dates, d);
        for g = 1:numel(dd.games)
            gm = elem(dd.games, g);
            % 只收已完賽
            if ~strcmp(gm.status.codedGameState, 'F'), continue; end
            aw = gm.teams.away;  hm = gm.teams.home;
            % 沒有比分的不收
            if ~isfield(aw, 'score') || ~isfield(hm, 'score'), continue; end
            if isempty(aw.score) || isempty(hm.score), continue; end
            m = m + 1;
            pk(m)  = gm.gamePk;
            dt{m}  = gm.officialDate;
            aId(m) = aw.team.id;   aNm{m} = aw.team.name;   aSc(m) = aw.score;
            hId(m) = hm.team.id;   hNm{m} = hm.team.name;   hSc(m) = hm.score;
        end
    end
    pk = pk(1:m); dt = dt(1:m);
    aId = aId(1:m); hId = hId(1:m); aNm = aNm(1:m); hNm = hNm(1:m);
    aSc = aSc(1:m); hSc = hSc(1:m);

    % 依 gamePk 去重（保改期／續賽的比賽可能出現在多個日期）
    [~, keep] = unique(pk);
    % 去掉平手
    keep = keep(aSc(keep) ~= hSc(keep));
    % 依日期、gamePk 排序，方便人工檢視
    [~, ord] = sortrows([datenum(dt(keep), 'yyyy-mm-dd'), pk(keep)]);
    keep = keep(ord);

    % 對應球隊縮寫（查不到就用隊名前三個字母）
    [tfA, locA] = ismember(aId, abbrId);
    [tfH, locH] = ismember(hId, abbrId);

    fid = fopen(csvFile, 'w');
    if fid < 0, error('無法寫入 %s', csvFile); end
    fprintf(fid, 'gamePk,date,away_id,away_abbr,away_name,home_id,home_abbr,home_name,away_score,home_score\n');
    for i = keep'
        if tfA(i), aAb = abbrStr{locA(i)}; else, aAb = upper(aNm{i}(1:3)); end
        if tfH(i), hAb = abbrStr{locH(i)}; else, hAb = upper(hNm{i}(1:3)); end
        fprintf(fid, '%d,%s,%d,%s,%s,%d,%s,%s,%d,%d\n', ...
            pk(i), dt{i}, aId(i), aAb, aNm{i}, hId(i), hAb, hNm{i}, aSc(i), hSc(i));
    end
    fclose(fid);
    fprintf('已寫入快取：%s（%d 場）\n', csvFile, numel(keep));
end

function x = elem(arr, i)
% jsondecode 依欄位是否一致，可能回傳 struct 陣列或 cell 陣列；統一取出第 i 個元素
    if iscell(arr), x = arr{i}; else, x = arr(i); end
end

function games = read_games_csv(csvFile)
% 讀取比賽快取 CSV（不依賴 readtable，以 textscan 解析）
    fid = fopen(csvFile, 'r');
    if fid < 0, error('無法開啟 %s', csvFile); end
    C = textscan(fid, '%f %s %f %s %s %f %s %s %f %f', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    games.gamePk    = C{1};
    games.date      = C{2};
    games.awayId    = C{3};
    games.awayAbbr  = C{4};
    games.awayName  = C{5};
    games.homeId    = C{6};
    games.homeAbbr  = C{7};
    games.homeName  = C{8};
    games.awayScore = C{9};
    games.homeScore = C{10};
end

function write_matrix_csv(csvFile, M, labels, fmt)
% 把方陣存成 CSV：第一列與第一欄是球隊縮寫
    fid = fopen(csvFile, 'w');
    if fid < 0, error('無法寫入 %s', csvFile); end
    fprintf(fid, 'from\\to');
    fprintf(fid, ',%s', labels{:});
    fprintf(fid, '\n');
    for i = 1:size(M, 1)
        fprintf(fid, '%s', labels{i});
        fprintf(fid, [',' fmt], M(i, :));
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function P = build_transition(A)
% 列正規化得到列隨機矩陣 P；沒有出邊（全勝）的列改為均勻分布
    n       = size(A, 1);
    rowSum  = sum(A, 2);
    P       = zeros(n);
    hasOut  = rowSum > 0;
    P(hasOut, :)  = A(hasOut, :) ./ repmat(rowSum(hasOut), 1, n);
    P(~hasOut, :) = 1 / n;
end

function [w, wHome, wAway] = game_weights(margin, awayWon, weightMode, homeMode, homePct)
% 每場比賽的邊權重 = 基本權重 × 主客場係數
%   基本權重：'win' 每勝 1、'margin' 分差、'log' log(1 + 分差)
%   主客場係數：'none' 都是 1；'data' 用全聯盟主場勝率 h；數值則直接當 h
%     主場勝 × 1/(2h)、客場勝 × 1/(2(1-h))  （反機率加權，平均係數為 1）
    switch lower(weightMode)
        case 'win',    base = ones(size(margin));
        case 'margin', base = margin;
        case 'log',    base = log(1 + margin);
        otherwise,     error('未知的 WEIGHT_MODE：%s（可用 win / margin / log）', weightMode);
    end
    if ischar(homeMode)
        switch lower(homeMode)
            case 'none', h = 0.5;
            case 'data', h = homePct;
            otherwise,   error('未知的 HOME_MODE：%s（可用 none / data / 數值）', homeMode);
        end
    else
        h = homeMode;
    end
    wHome = 1 / (2 * h);
    wAway = 1 / (2 * (1 - h));
    w = base * wHome;
    w(awayWon) = base(awayWon) * wAway;
end

function d = build_damping(losses, c)
% 每隊的阻尼值 d_i = L_i / (L_i + c·mean(L))，介於 0 與 1 之間
%   L_i 越大（輸越多場、出邊越多、證據越充分）→ d_i 越大 → 越相信比賽結果
%   L_i = 0（全勝）→ d_i = 0 → 該列完全隨機跳轉（P 的那列本來也是均勻分布）
    L = losses(:);
    d = L ./ (L + c * mean(L));
end

function G = google_matrix(P, d)
% Google 矩陣 G = D·P + (I - D)·E，D = diag(d) 為阻尼矩陣，E = 1·1'/n 為均勻跳轉
%   第 i 列：G(i,:) = d_i·P(i,:) + (1 - d_i)/n，每列和仍為 1
    n = size(P, 1);
    D = diag(d(:));
    G = D * P + (eye(n) - D) * ones(n) / n;
end

function [pr, it, delta] = pagerank_power(P, d, tol, maxIter)
% 冪迭代：π(k+1)' = π(k)' G，直到 L1 變化量 < tol
    n  = size(P, 1);
    G  = google_matrix(P, d);
    v  = ones(1, n) / n;
    delta = Inf;
    for it = 1:maxIter
        vNew  = v * G;
        vNew  = vNew / sum(vNew);       % 抵銷浮點誤差累積
        delta = sum(abs(vNew - v));
        v     = vNew;
        if delta < tol, break; end
    end
    if delta >= tol
        warning('冪迭代在 %d 次內未收斂（L1 變化量 = %.2e）', maxIter, delta);
    end
    pr = v(:);
end

function pr = pagerank_eig(P, d)
% 直接求 G' 對應特徵值 1 的特徵向量（即 G 的左特徵向量），正規化為機率分布
    G = google_matrix(P, d);
    [V, D] = eig(G');
    [~, k] = min(abs(diag(D) - 1));
    v  = real(V(:, k));
    pr = v / sum(v);
end

function r = competition_rank(x)
% 競賽式排名（由小到大；同值同名次，如 1,2,2,4）
    x = x(:);
    r = zeros(size(x));
    for i = 1:numel(x)
        r(i) = 1 + sum(x < x(i));
    end
end

function r = average_rank(x)
% 平均名次（同值取平均），供 Spearman 使用；取代 Statistics Toolbox 的 tiedrank
    x = x(:);
    r = zeros(size(x));
    for i = 1:numel(x)
        r(i) = sum(x < x(i)) + (sum(x == x(i)) + 1) / 2;
    end
end

function rho = spearman_corr(x, y)
% Spearman 等級相關 = 兩組平均名次的 Pearson 相關
    rx = average_rank(x);  ry = average_rank(y);
    rx = rx - mean(rx);    ry = ry - mean(ry);
    rho = sum(rx .* ry) / sqrt(sum(rx .^ 2) * sum(ry .^ 2));
end

function fetch_postseason(season, csvFile)
% 抓季後賽（F 外卡、D 分區系列賽、L 聯盟冠軍賽、W 世界大賽）已完賽的比賽，寫成 CSV 快取
    opts = weboptions('Timeout', 120, 'ContentType', 'text');
    url  = sprintf(['https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=F,D,L,W' ...
                    '&startDate=%d-01-01&endDate=%d-12-31'], season, season);
    sj   = jsondecode(webread(url, opts));
    rows = {};  pk = [];
    for d = 1:numel(sj.dates)
        dd = elem(sj.dates, d);
        for g = 1:numel(dd.games)
            gm = elem(dd.games, g);
            if ~strcmp(gm.status.codedGameState, 'F'), continue; end
            aw = gm.teams.away;  hm = gm.teams.home;
            if ~isfield(aw, 'score') || ~isfield(hm, 'score'), continue; end
            if isempty(aw.score) || isempty(hm.score) || aw.score == hm.score, continue; end
            if any(pk == gm.gamePk), continue; end          % 依 gamePk 去重
            pk(end + 1) = gm.gamePk; %#ok<AGROW>
            rows{end + 1} = sprintf('%d,%s,%s,%s,%d,%d,%s,%d,%s,%d,%d', gm.gamePk, ...
                gm.officialDate, gm.gameType, gm.seriesDescription, gm.gamesInSeries, ...
                aw.team.id, aw.team.name, hm.team.id, hm.team.name, aw.score, hm.score); %#ok<AGROW>
        end
    end
    fid = fopen(csvFile, 'w');
    if fid < 0, error('無法寫入 %s', csvFile); end
    fprintf(fid, 'gamePk,date,game_type,series,games_in_series,away_id,away_name,home_id,home_name,away_score,home_score\n');
    for i = 1:numel(rows)
        fprintf(fid, '%s\n', rows{i});
    end
    fclose(fid);
    fprintf('已寫入季後賽快取：%s（%d 場）\n', csvFile, numel(rows));
end

function post = read_post_csv(csvFile)
% 讀取季後賽快取 CSV
    fid = fopen(csvFile, 'r');
    if fid < 0, error('無法開啟 %s', csvFile); end
    C = textscan(fid, '%f %s %s %s %f %f %s %f %s %f %f', ...
        'Delimiter', ',', 'HeaderLines', 1, 'Whitespace', '');
    fclose(fid);
    post.gamePk        = C{1};
    post.date          = C{2};
    post.gameType      = C{3};
    post.series        = C{4};
    post.gamesInSeries = C{5};
    post.awayId        = C{6};
    post.homeId        = C{8};
    post.awayScore     = C{10};
    post.homeScore     = C{11};
end

function [pWin, kBest] = series_outcome(p, need)
% 單場勝率為 p、先拿 need 勝者晉級：回傳贏下系列賽的機率，
% 以及「預測贏家」最可能讓對手拿到幾勝（例如 4-2 的 2）
    k  = 0:need - 1;
    c  = zeros(size(k));
    for i = 1:numel(k), c(i) = nchoosek(need - 1 + k(i), k(i)); end
    pa = c .* p ^ need .* (1 - p) .^ k;      % 我方以 need-k 晉級
    pb = c .* (1 - p) ^ need .* p .^ k;      % 對手以 need-k 晉級
    pWin = sum(pa);
    if pWin >= 0.5, [~, i] = max(pa); else, [~, i] = max(pb); end
    kBest = k(i);
end

function W = series_win_matrix(Q, need)
% 把單場勝率矩陣 Q 換成「先拿 need 勝」的系列賽勝率矩陣
    W = zeros(size(Q));
    for k = 0:need - 1
        W = W + nchoosek(need - 1 + k, k) * Q .^ need .* (1 - Q) .^ k;
    end
end
