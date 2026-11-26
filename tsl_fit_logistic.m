function w = tsl_fit_logistic(P, y)
% TSL_FIT_LOGISTIC  以牛頓法擬合邏輯斯迴歸 y ~ σ( Σ w_k·logit(P(:,k)) + w_end )
%
%   w = tsl_fit_logistic(P, y)
%
%   P  m × K，每一欄是一組機率（會先轉成 logit 當特徵）
%   y  m × 1，0 / 1
%   w  (K+1) × 1，前 K 個是各欄的係數，最後一個是截距
%
%   K = 1 時就是校準 p = σ(a·logit(pRaw) + b)，w = [a; b]。
%   K = 2 時可以同時放模型與市場的機率，看模型在市場之外還有沒有資訊。
    P = min(max(P, 1e-6), 1 - 1e-6);
    F = [log(P ./ (1 - P)), ones(size(P, 1), 1)];
    K = size(F, 2);
    w = [ones(K - 1, 1) / (K - 1); 0];
    for it = 1:50
        p    = 1 ./ (1 + exp(-F * w));
        H    = F' * (F .* repmat(p .* (1 - p), 1, K)) + 1e-9 * eye(K);
        step = H \ (F' * (y - p));
        w    = w + step;
        if max(abs(step)) < 1e-10, break; end
    end
end
