function ll = tsl_logloss(p, y)
% TSL_LOGLOSS  平均 log loss：−mean( y·ln p + (1−y)·ln(1−p) )，越低越準；丟銅板 = ln 2 ≈ 0.693
    p  = min(max(p, 1e-12), 1 - 1e-12);
    ll = -mean(y .* log(p) + (1 - y) .* log(1 - p));
end
