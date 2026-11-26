// 台灣運彩 MLB「不讓分」賠率下載（在瀏覽器主控台執行）
//
// 運彩網站有 Cloudflare 機器人驗證，Python 直接呼叫會被擋（HTTP 403），
// 所以改在你自己的瀏覽器裡、用網站本身的 API 抓，抓完自動下載 tsl_odds_YYYY-MM-DD.csv。
//
// 用法：
//   1. 開 https://www-talo-ssb-pr.sportslottery.com.tw/  （運彩投注頁本體，不是外層 www.sportslottery.com.tw）
//   2. 開發者工具 → Console，貼上整段程式碼後按 Enter
//   3. 把下載的 CSV 放進專案資料夾（或用 --odds 指定路徑），執行 python3 tsl_predict.py
//
// 欄位：event_id, start（台灣時間）, away, home, away_odds, home_odds（十進位賠率，含本金）
(async () => {
  const NODE = '36330.1';                       // 棒球 → 美國 → 美國職棒
  const get = (type, id) => fetch('/services/content/get', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contentId: { type, id }, clientContext: { language: 'ZH', ipAddress: '0.0.0.0' } }),
  }).then(r => r.json());
  const dec = s => 1 + Number(s.currentpriceup) / Number(s.currentpricedown);

  const nav = await get('boNavigationList', '1355/' + NODE);
  const groups = (nav.data?.marketgroups || []).map(g => g.idfwmarketgroup);
  const rows = [];
  for (const g of groups) {
    const eg = await get('eventGroup', g);
    for (const ev of eg.data?.events || []) {
      const m = (ev.markets || []).find(m => m.name.trim() === '不讓分');
      if (!m) continue;
      const a = m.selections.find(s => s.hadvalue === 'A');
      const h = m.selections.find(s => s.hadvalue === 'H');
      if (!a || !h) continue;
      rows.push([ev.idfoevent, ev.tsstart, ev.participantname_away.trim(), ev.participantname_home.trim(),
                 dec(a).toFixed(3), dec(h).toFixed(3)]);
    }
  }
  if (!rows.length) { console.warn('目前沒有開賣的美國職棒「不讓分」賽事。'); return; }
  const csv = ['event_id,start,away,home,away_odds,home_odds', ...rows.map(r => r.map(x => `"${x}"`).join(','))].join('\n');
  const day = new Date().toLocaleDateString('sv-SE', { timeZone: 'Asia/Taipei' });
  const link = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv' })),
    download: `tsl_odds_${day}.csv`,
  });
  link.click();
  console.log(`已下載 ${rows.length} 場：tsl_odds_${day}.csv`);
})();
