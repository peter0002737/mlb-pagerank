function tsl_fetch_games(season, csvFile)
% TSL_FETCH_GAMES  從 MLB Stats API 抓整季例行賽，寫成 mlb_games_YYYY.csv 快取
%
%   與 mlb_pagerank.m 的 fetch_games 相同格式：只留已完賽 (codedGameState = 'F')、
%   依 gamePk 去重、去掉平手，依日期排序。
    opts = weboptions('Timeout', 120, 'ContentType', 'text');

    % --- 球隊縮寫 ---
    tj = jsondecode(webread(sprintf( ...
        'https://statsapi.mlb.com/api/v1/teams?sportId=1&season=%d', season), opts));
    nT = numel(tj.teams);
    abbrId = zeros(nT, 1);  abbrStr = cell(nT, 1);
    for i = 1:nT
        t = elem(tj.teams, i);
        abbrId(i) = t.id;  abbrStr{i} = t.abbreviation;
    end

    % --- 賽程 ---
    sj = jsondecode(webread(sprintf(['https://statsapi.mlb.com/api/v1/schedule?sportId=1' ...
        '&gameType=R&startDate=%d-01-01&endDate=%d-12-31'], season, season), opts));
    nMax = 0;
    for d = 1:numel(sj.dates)
        dd = elem(sj.dates, d);                  % MATLAB 不能直接對函式回傳值取欄位
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
            if ~strcmp(gm.status.codedGameState, 'F'), continue; end      % 只收已完賽
            aw = gm.teams.away;  hm = gm.teams.home;
            if ~isfield(aw, 'score') || ~isfield(hm, 'score'), continue; end
            if isempty(aw.score) || isempty(hm.score), continue; end
            m = m + 1;
            pk(m) = gm.gamePk;  dt{m} = gm.officialDate;
            aId(m) = aw.team.id;  aNm{m} = aw.team.name;  aSc(m) = aw.score;
            hId(m) = hm.team.id;  hNm{m} = hm.team.name;  hSc(m) = hm.score;
        end
    end
    if m == 0
        error('%d 年還沒有任何已完賽的例行賽。', season);
    end
    pk = pk(1:m); dt = dt(1:m); aId = aId(1:m); hId = hId(1:m);
    aNm = aNm(1:m); hNm = hNm(1:m); aSc = aSc(1:m); hSc = hSc(1:m);

    [~, keep] = unique(pk);                      % 依 gamePk 去重（改期、續賽）
    keep = keep(aSc(keep) ~= hSc(keep));         % 去掉平手
    [~, ord] = sortrows([datenum(dt(keep), 'yyyy-mm-dd'), pk(keep)]);
    keep = keep(ord);

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
% jsondecode 可能回傳 struct 陣列或 cell 陣列；統一取出第 i 個元素
    if iscell(arr), x = arr{i}; else, x = arr(i); end
end
