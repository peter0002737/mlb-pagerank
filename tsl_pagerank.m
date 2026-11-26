function pr = tsl_pagerank(AH, AA, W, L, alphas)
% TSL_PAGERANK  依勝率跳轉的 PageRank（個人化 PageRank），可一次算多個 α
%
%   pr = tsl_pagerank(AH, AA, W, L, alphas)
%
%   輸入
%     AH(i,j)  第 i 隊在客場輸給主隊 j 的場數（主場勝）
%     AA(i,j)  第 i 隊在主場輸給客隊 j 的場數（客場勝）
%     W, L     每隊目前的勝場、敗場數（n×1 或 1×n）
%     alphas   阻尼係數向量（沿著比賽結果走的機率）
%   輸出
%     pr       numel(alphas) × n，每一列是一個 α 的 PageRank（總和為 1）
%
%   模型（見 README）
%     邊權重：主場勝 × 1/(2h)、客場勝 × 1/(2(1−h))，h = 全聯盟主場勝率
%     P = A 每列正規化（全勝、沒有出邊的隊伍改成均勻分布）
%     s_i = (W_i + 1) / (W_i + L_i + 2)，v = s / sum(s)（勝率加 1 平滑）
%     G = α·P + (1−α)·1·vᵀ，解 πᵀG = πᵀ、sum(π) = 1

    n  = numel(W);
    W  = W(:);  L = L(:);

    % --- 主客場反機率加權的鄰接矩陣 A(輸家, 贏家) ---
    nh = sum(AH(:));  na = sum(AA(:));
    h  = nh / max(nh + na, eps);
    if h <= 0 || h >= 1
        A = AH + AA;                         % 還沒有比賽或只有一種結果時不加權
    else
        A = AH / (2 * h) + AA / (2 * (1 - h));
    end

    % --- 轉移矩陣 P ---
    rs = sum(A, 2);
    P  = ones(n) / n;
    ok = rs > 0;
    P(ok, :) = A(ok, :) ./ repmat(rs(ok), 1, n);

    % --- 依平滑勝率的跳轉分布 v ---
    s = (W + 1) ./ (W + L + 2);
    v = (s / sum(s))';                       % 1×n

    % --- 每個 α 解一次線性方程組 (Gᵀ − I)π = 0，最後一列換成 sum(π) = 1 ---
    pr  = zeros(numel(alphas), n);
    rhs = zeros(n, 1);  rhs(end) = 1;
    for k = 1:numel(alphas)
        G = alphas(k) * P + (1 - alphas(k)) * repmat(v, n, 1);
        M = G' - eye(n);
        M(end, :) = 1;
        pr(k, :) = (M \ rhs)';
    end
end
