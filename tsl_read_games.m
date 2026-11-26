function games = tsl_read_games(csvFile)
% TSL_READ_GAMES  讀取 mlb_games_YYYY.csv 比賽快取（與 mlb_pagerank.m 的 read_games_csv 相同）
%
%   games 為 struct，欄位：gamePk, date（cellstr, yyyy-mm-dd）, awayId, awayAbbr,
%   awayName, homeId, homeAbbr, homeName, awayScore, homeScore
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
