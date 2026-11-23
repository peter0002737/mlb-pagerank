# MLB PageRank

用 PageRank 為 MLB 球隊排名，並研究阻尼係數該怎麼調。以 2021–2026 六個球季（13,639 場）逐日滾動回測，比較文獻中的幾種阻尼調整方式。

> **注意**：主程式 `mlb_pagerank.m` 目前仍使用原本的阻尼矩陣（c = 1）。下面「依勝率跳轉」的模型與其他阻尼方法，目前只實作在 Python 回測程式中。

## 符號

| 符號 | 意思 |
|---|---|
| $n$ | 球隊數（30） |
| $W_i,\ L_i$ | 第 $i$ 隊目前的勝場、敗場數 |
| $h$ | 全聯盟主場勝率 |
| $A$ | 加權鄰接矩陣，$A(i,j)$ = $i$ 輸給 $j$ 的累計權重（邊：輸家 → 贏家） |
| $P$ | 轉移矩陣（$A$ 每列正規化） |
| $s_i$ | 第 $i$ 隊的平滑勝率 |
| $v$ | 跳轉分布（依勝率分配） |
| $\alpha$ | 阻尼係數：沿著比賽結果走的機率 |
| $G$ | Google 矩陣 |
| $\pi$ | PageRank（穩態分布） |

## 模型：依勝率跳轉的 PageRank（個人化 PageRank）

經典 PageRank 以機率 $1-\alpha$ **均勻**跳到任一隊；這裡改成依**勝率**決定跳到哪一隊（topic-specific / personalized PageRank）。

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

2021–2026 回測的最佳值為 $\alpha = 0.15$（留一球季交叉驗證，六次中四次挑到 0.15、兩次 0.2）。

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
| **依勝率跳轉** | $G = \alpha P + (1-\alpha)\mathbf{1}v^{\mathsf T}$ | $\alpha = 0.15$ | **0.68351** | 55.5% |
| 兩參數自適應 | $d_i = L_i^a / (L_i^a + k^a)$ | $k = 12,\ a = 3$ | 0.68382 | 55.4% |
| 自適應 | $d_i = L_i / (L_i + k)$ | $k = 6$ | 0.68413 | 55.2% |
| 原阻尼矩陣 | $d_i = L_i / (L_i + c\cdot\overline{L})$ | $c = 0.3$ | 0.68518 | 55.3% |
| 隨機漫步者 | 以機率 $p$ 跟贏家、$1-p$ 跟輸家 | $p = 0.75$ | 0.68520 | 55.5% |
| 單一 α（均勻跳轉） | $d_i = \alpha$ | $\alpha = 0.75$ | 0.68563 | 55.3% |
| 對照：平滑勝率（不用網絡） | $\pi = v$（即 $\alpha = 0$） | — | 0.68361 | 55.4% |

除了依勝率跳轉之外，其餘方法都用均勻跳轉 $G = DP + (I-D)E$，$D = \mathrm{diag}(d_i)$，$E$ 的每個元素為 $1/n$。

**解讀**

- 依勝率跳轉的 log loss 最低，但只用平滑勝率（$\alpha = 0$）就有 0.68361，加入比賽網絡只再降 0.0001：優勢主要來自勝率本身。
- 只用比賽網絡（均勻跳轉）時，最佳是兩參數自適應阻尼。它讓阻尼在約 40 場附近快速上升：每隊打 20 場 $d \approx 0.37$、40 場 $0.82$、80 場 $0.97$。
- 所有方法的命中率都在 55.2–55.5%，調阻尼影響的是機率準不準，不是猜不猜得到勝隊。

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
