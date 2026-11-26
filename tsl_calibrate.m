function p = tsl_calibrate(pRaw, a, b)
% TSL_CALIBRATE  把主隊原始勝率校準成實際勝率
%
%   p = tsl_calibrate(pRaw, a, b)
%
%   pRaw = π_主 / (π_主 + π_客)（PageRank 算出來的主隊勝率，沒有主場優勢）
%   p    = σ( a · logit(pRaw) + b )
%          b > 0 代表主場優勢；a < 1 代表原模型太自信、a > 1 代表太保守
    pRaw = min(max(pRaw, 1e-6), 1 - 1e-6);
    p    = 1 ./ (1 + exp(-(a * log(pRaw ./ (1 - pRaw)) + b)));
end
