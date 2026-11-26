# MLB PageRank

用 PageRank 為 MLB 球隊排名，並研究阻尼係數該怎麼調。以 2021–2026 六個球季（13,639 場）逐日滾動回測，比較文獻中的幾種阻尼調整方式。

> **注意**：主程式 `mlb_pagerank.m` 目前仍使用原本的阻尼矩陣（c = 1）。下面「依勝率跳轉」的模型與其他阻尼方法，目前只實作在 Python 回測程式中。

## 符號

| 符號 | 意思 |
|---|---|
| $`n`$ | 球隊數（30） |
| $`W_i,\ L_i`$ | 第 $`i`$ 隊目前的勝場、敗場數 |
| $`h`$ | 全聯盟主場勝率 |
| $`A`$ | 加權鄰接矩陣，$`A(i,j)`$ = $`i`$ 輸給 $`j`$ 的累計權重（邊：輸家 → 贏家） |
| $`P`$ | 轉移矩陣（$`A`$ 每列正規化） |
| $`s_i`$ | 第 $`i`$ 隊的平滑勝率 |
| $`v`$ | 跳轉分布（依勝率分配） |
| $`\alpha`$ | 阻尼係數：沿著比賽結果走的機率 |
| $`G`$ | Google 矩陣 |
| $`\pi`$ | PageRank（穩態分布） |

## 模型：依勝率跳轉的 PageRank（個人化 PageRank）

經典 PageRank 以機率 $`1-\alpha`$ **均勻**跳到任一隊；這裡改成依**勝率**決定跳到哪一隊（topic-specific / personalized PageRank）。

**1. 邊權重**：每勝 1，主客場反機率加權

$$
w = \begin{cases} \dfrac{1}{2h} & \text{主場勝} \\[4pt] \dfrac{1}{2(1-h)} & \text{客場勝} \end{cases}
$$

**2. 轉移矩陣**

$$
P(i,j) = \frac{A(i,j)}{\sum_k A(i,k)}
$$

**3. 跳轉分布**：勝率加 1 平滑（Laplace smoothing）後正規化

$$
s_i = \frac{W_i + 1}{W_i + L_i + 2}, \qquad v_i = \frac{s_i}{\sum_j s_j}
$$

**4. Google 矩陣與 PageRank**

$$
G = \alpha\,P + (1-\alpha)\,\mathbf{1}\,v^{\mathsf T}, \qquad \pi^{\mathsf T} G = \pi^{\mathsf T}, \quad \sum_i \pi_i = 1
$$

2021–2026 回測的最佳值為 $`\alpha = 0.15`$（留一球季交叉驗證，六次中四次挑到 0.15、兩次 0.2）。

**5. 單場勝率**

$$
p(A \text{ 勝 } B) = \frac{\pi_A}{\pi_A + \pi_B}
$$

**評估指標**

$$
\text{log loss} = -\frac{1}{N} \sum_{\text{每場}} \ln p(\text{實際勝隊})
$$

命中率 = 機率 > 0.5 的一方真的獲勝的比例。

## 阻尼調整方法比較（2021–2026）

逐日滾動預測（只用當天以前的比賽），兩隊都打滿 10 場才計分；參數用留一球季交叉驗證挑選。

| 方法 | 阻尼公式 | 最佳參數 | 樣本外 log loss | 命中率 |
|---|---|---|---|---|
| **依勝率跳轉** | $`G = \alpha P + (1-\alpha)\mathbf{1}v^{\mathsf T}`$ | $`\alpha = 0.15`$ | **0.68351** | 55.5% |
| 兩參數自適應 | $`d_i = L_i^a / (L_i^a + k^a)`$ | $`k = 12,\ a = 3`$ | 0.68382 | 55.4% |
| 自適應 | $`d_i = L_i / (L_i + k)`$ | $`k = 6`$ | 0.68413 | 55.2% |
| 原阻尼矩陣 | $`d_i = L_i / (L_i + c\cdot\overline{L})`$ | $`c = 0.3`$ | 0.68518 | 55.3% |
| 隨機漫步者 | 以機率 $`p`$ 跟贏家、$`1-p`$ 跟輸家 | $`p = 0.75`$ | 0.68520 | 55.5% |
| 單一 α（均勻跳轉） | $`d_i = \alpha`$ | $`\alpha = 0.75`$ | 0.68563 | 55.3% |
| 對照：平滑勝率（不用網絡） | $`\pi = v`$（即 $`\alpha = 0`$） | — | 0.68361 | 55.4% |

除了依勝率跳轉之外，其餘方法都用均勻跳轉 $`G = DP + (I-D)E`$，$`D = \mathrm{diag}(d_i)`$，$`E`$ 的每個元素為 $`1/n`$。

**解讀**

- 依勝率跳轉的 log loss 最低，但只用平滑勝率（$`\alpha = 0`$）就有 0.68361，加入比賽網絡只再降 0.0001：優勢主要來自勝率本身。
- 只用比賽網絡（均勻跳轉）時，最佳是兩參數自適應阻尼。它讓阻尼在約 40 場附近快速上升：每隊打 20 場 $`d \approx 0.37`$、40 場 $`0.82`$、80 場 $`0.97`$。
- 所有方法的命中率都在 55.2–55.5%，調阻尼影響的是機率準不準，不是猜不猜得到勝隊。

## 運彩預測（MLB 不讓分）

把依勝率跳轉的 PageRank 接到台灣運彩的「不讓分」賠率，算出每一邊的期望值。

**1. 校準**：下注看的是 $`\text{EV} = p \times \text{賠率} - 1`$，所以 $`p`$ 要準。原模型沒有主場優勢，因此再加一層邏輯斯校準：

$$
p_{\text{主}} = \sigma\!\left(a \cdot \operatorname{logit}\frac{\pi_{\text{主}}}{\pi_{\text{主}} + \pi_{\text{客}}} + b\right)
$$

2016–2026 逐日滾動、留一球季交叉驗證（23,462 場）：

| | 樣本外 log loss |
|---|---|
| 未校準（$`\alpha = 0.15`$，無主場） | 0.68364 |
| **校準後**（$`\alpha = 0.2,\ a = 1.007,\ b = +0.136`$） | **0.68149** |
| 丟銅板 | 0.69315 |

$`b = 0.136`$ 表示實力相當時主隊勝率約 53.4%；$`a \approx 1`$ 表示原模型的信心程度本來就差不多準。

| 被看好方預測勝率 | 場數 | 實際勝率 | 不賠錢所需最低賠率 |
|---|---|---|---|
| 50.0–52.5% | 6315 | 51.9% | 1.93 |
| 55.0–57.5% | 4541 | 56.2% | 1.78 |
| 60.0–62.5% | 1902 | 61.7% | 1.62 |
| 65% 以上 | 1176 | 66.8% | 1.50 |

**2. 抓賠率**：運彩網站有 Cloudflare 機器人驗證，程式直接呼叫 API 會得到 403，所以改在自己的瀏覽器執行 `tsl_odds_snippet.js`：開 `https://www-talo-ssb-pr.sportslottery.com.tw/`，在開發者工具 Console 貼上執行，會下載 `tsl_odds_YYYY-MM-DD.csv`。十進位賠率 = 1 + 分數賠率（API 的 `currentpriceup / currentpricedown`）。

**3. 預測**：

```matlab
tsl_calibration_backtest                 % 先跑一次，產生 tsl_calibration.csv（已附，可略過）
tsl_predict                              % 自動找專案或「下載」資料夾裡最新的 tsl_odds_*.csv
tsl_predict('tsl_odds_2027-04-01.csv')   % 指定檔案
tsl_predict(檔案, false, 0.05)           % 不更新賽程、只標示 EV > 5% 的選項
```

輸出每場兩邊的模型勝率、運彩隱含勝率（去掉抽水後）、賠率與 EV，存成 `tsl_predictions_YYYY-MM-DD.csv`。本季賽程會自動從 MLB Stats API 更新，只用開賽日以前的比賽。

**限制**

- 運彩抽水約 15–17%（兩邊隱含機率加總約 1.17），而模型 log loss 只比丟銅板低 0.012；長期要贏，通常需要比市場更準的資訊。
- 模型只看勝負，不知道先發投手、傷兵、休息日，這些都是市場已經反映的資訊。EV 為正可能只是模型漏看了什麼。
- 抓賠率的程式在 2026-11（休賽期）用籃球賽事驗證過；MLB 開賣後，若玩法名稱不是「不讓分」或隊名對不上，需要調整 `tsl_odds_snippet.js` 或 `tsl_predict.m` 中 `team_index` 的 `KEY`。

## 國際盤回測（模型 vs 博彩公司）

**資料**：`tsl_fetch_espn_odds(年份)` 從 ESPN 公開 API 抓每場例行賽的收盤獨贏賠率（非官方文件化的 API，僅供個人研究）。每場收錄數家博彩公司，2016 年有 Pinnacle、5Dimes、Bovada 等十餘家，2026 年只有 DraftKings。排除場中賠率、consensus 等非盤口資料，以及主客標反的異常值，再取各家中位數與最佳賠率。以「日期＋兩隊＋比分」對應 `mlb_games`，雙重賽也不會配錯；可中斷續抓。

**回測**：`tsl_market_backtest` 回答三個問題：

1. **誰比較準**：模型（留一季校準）與市場（去水後勝率中位數）的 log loss。
2. **模型有沒有市場不知道的資訊**：$`y \sim \sigma(w_{\text{模型}}\operatorname{logit}p_{\text{模型}} + w_{\text{市場}}\operatorname{logit}p_{\text{市場}} + c)`$，$`w_{\text{模型}}`$ 顯著大於 0 才代表模型有額外資訊。
3. **照模型下注賺不賺**：每場挑 EV 較大的一邊，EV 超過門檻就押 1 單位，計算 ROI 與 95% 信賴區間。

目前賠率資料只完整下載了 2016 與 2026 兩季（2017 下載到一半），`tsl_market_backtest` 的多季回測尚未執行。以下是 2026 球季（2,277 場，DraftKings 收盤賠率，平均抽水 4.13%）的結果。

### 2026 逐日下注模擬（`tsl_daily_bet_sim`）

模型：依兩隊中較少的已賽場數查 `damping_by_games.csv` 決定 α 的經典 PageRank（分段阻尼），主場校準用 2016–2025 擬合（$`a = 1.288,\ b = +0.138`$）。log loss：模型 0.68756、市場 0.67876、丟銅板 0.69315。

| 下注方法 | 注數 | 損益（每注 100 元） | ROI（95% 信賴區間） | 押中率 |
|---|---|---|---|---|
| 正 EV、固定金額 | 1,694 | −8,819 | −5.21%（−10.44% ～ +0.03%） | 43.7% |
| 正 EV、賠率 < 2.0 | 530 | −4,364 | −8.23%（−16.08% ～ −0.39%） | 50.0% |
| 正 EV、賠率 < 1.8 | 173 | −1,684 | −9.74%（−22.56% ～ +3.09%） | 52.6% |
| 正 EV、只押熱門 | 388 | −1,900 | −4.90%（−13.87% ～ +4.08%） | 52.8% |
| 正 EV、1/4 凱利（起始 10,000 元） | 1,490 | −9,934（期末 66 元） | −3.24%（−9.46% ～ +1.70%） | 44.5% |
| 對照：每場押主隊 | 2,277 | −10,340 | −4.54%（−8.35% ～ −0.74%） | 52.7% |
| 對照：每場押熱門 | 2,277 | −6,774 | −2.98%（−6.45% ～ +0.50%） | 57.2% |

所有用模型挑選的方法都虧損；不看模型、每場押熱門反而虧最少（熱門–冷門偏差）。凱利公式在模型沒有優勢時會放大虧損。

### 加入先發投手與近期加權（`tsl_pitcher_season`）

先發投手取 MLB Stats API 的賽前預定先發，能力用本季到前一天為止、向聯盟平均收縮（30 局）的 ERA* 或 FIP*。所有係數只用 2025 擬合，2026 為樣本外測試；近期加權半衰期 60 天。

| 模型（2025 擬合） | 2026 log loss | 與市場平均差距 | 模型增量資訊 $`w`$（z） | 下注 ROI（EV > 0） |
|---|---|---|---|---|
| 市場 | **0.67876** | — | — | — |
| 基本（依勝率跳轉 PageRank） | 0.68829 | 5.1 點 | −0.144（−0.5） | −4.91% |
| 近期加權 | 0.68860 | 5.0 點 | −0.201（−0.7） | −6.54% |
| 基本＋先發 ERA（每點 0.090） | **0.68430** | 4.6 點 | +0.314（+1.1） | −2.99% |
| 基本＋先發 FIP（每點 0.129） | 0.68553 | 4.3 點 | −0.089（−0.3） | −5.04% |
| 近期＋先發 FIP | 0.68581 | 4.2 點 | −0.146（−0.5） | −4.09% |

**解讀**

- 加入先發投手讓模型與市場的 log loss 差距從 0.0095 縮小到 0.0055（約四成），是目前最有效的改進；近期加權沒有幫助。
- 擬合出的 ERA 係數（0.090）只有畢氏勝率理論近似（0.254）的三分之一。
- 4 月時各模型與市場一樣準（市場 0.6979、基本＋ERA 0.6964），5 月以後市場明顯領先。
- 模型對市場的增量資訊都不顯著（|z| < 2），下注報酬皆為負：只用勝負網絡與先發 ERA，還贏不了 4% 抽水的國際盤，更不可能贏約 17% 抽水的運彩。

## 檔案

| 檔案 | 用途 |
|---|---|
| `mlb_pagerank.m` | 主程式（MATLAB，不需 Toolbox）：抓賽程、算 PageRank、排名、敏感度分析、季後賽回測與奪冠機率 |
| `damping_backtest.py` | 阻尼回測基本版：2021–2025 單一 α 與阻尼矩陣 c，含資料下載 |
| `damping_stage_backtest.py` | 分段阻尼：開季、季中、季末各自的最佳阻尼 |
| `damping_by_games.py` | 每 10 場一組的最佳阻尼曲線 |
| `damping_methods_backtest.py` | 六種阻尼調整方法比較（2021–2026），本頁表格的來源 |
| `damping_backtest_results.csv` | `damping_backtest.py` 的完整結果 |
| `damping_by_games.csv` | 每 10 場一組的最佳阻尼 |
| `damping_methods_results.csv` | 六種方法的最佳參數與各季 log loss |
| `tsl_calibration_backtest.m` | 運彩用的勝率校準回測（2016–2026），產生 `tsl_calibration.csv` |
| `tsl_calibration.csv` | 校準參數 α、a、b |
| `tsl_odds_snippet.js` | 在瀏覽器 Console 執行，下載運彩 MLB 不讓分賠率 |
| `tsl_predict.m` | 模型勝率 vs 運彩賠率，計算期望值 |
| `tsl_pagerank.m`、`tsl_calibrate.m` | 依勝率跳轉的 PageRank、勝率校準（上面兩支共用） |
| `tsl_fetch_games.m`、`tsl_read_games.m` | 下載／讀取 `mlb_games_YYYY.csv` 賽程快取 |
| `tsl_rolling_probs.m` | 逐日滾動計算每場主隊原始勝率（校準與國際盤回測共用） |
| `tsl_fit_logistic.m`、`tsl_logloss.m` | 邏輯斯迴歸（牛頓法）、log loss |
| `tsl_fetch_espn_odds.m` | 從 ESPN 下載某季國際盤獨贏賠率，存成 `espn_odds_YYYY.csv` |
| `tsl_market_backtest.m` | 模型 vs 國際盤回測，結果存成 `tsl_market_backtest.csv` |
| `tsl_daily_bet_sim.m` | 單季逐日下注模擬：分段阻尼 PageRank，比較固定金額、限制賠率、只押熱門、1/4 凱利等方法 |
| `tsl_pitcher_trial.m` | 小規模試算：某一天的比賽加入先發投手與近期加權後的勝率變化 |
| `tsl_pitcher_season.m` | 整季檢驗先發投手（ERA*、FIP*）與近期加權：前一季擬合、下一季測試 |

## 執行

MATLAB：

```matlab
mlb_pagerank
```

Python（需要 numpy）：

```bash
pip install numpy
python3 damping_backtest.py          # 第一次執行會下載 2021–2025 賽程快取
python3 damping_methods_backtest.py  # 需要 mlb_games_2021–2026.csv
```

兩邊第一次執行時都會從 MLB Stats API 下載賽程並存成 CSV 快取；2026 年的快取由 `mlb_pagerank.m` 產生。

## 參考資料

- [Topic-specific PageRank（Stanford IR book）](https://nlp.stanford.edu/IR-book/html/htmledition/topic-specific-pagerank-1.html)
- [Gleich, PageRank Beyond the Web](https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202015%20-%20prbeyond.pdf)
- [Callaghan, Mucha & Porter, Random Walker Ranking for NCAA Division I-A Football](https://arxiv.org/abs/physics/0310148)
- [Effect of stepwise adjustment of damping factor upon PageRank（arXiv 2108.04150）](https://arxiv.org/pdf/2108.04150)
- [Limits of PageRank-based ranking methods in sports data（arXiv 2012.06366）](https://arxiv.org/abs/2012.06366)
- [PageRank and rank-reversal dependence on the damping factor（arXiv 1201.4787）](https://arxiv.org/pdf/1201.4787)
