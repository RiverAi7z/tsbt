# 可用数据清单

> 2026-11-19 实测。只列**已补全到最新**的数据：10 个标的 × 6 个数据集，时间 **2023-01-01 起**。其他标的和其他数据集（`trades`、`liquidationSnapshot` 等）没有补全，不在本文范围内，暂不使用。

## 范围

| 项 | 值 |
|---|---|
| 标的（10） | `BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT LTCUSDT`（均为 USDⓈ-M 永续） |
| 起点 | 2023-01-01 00:00 UTC |
| 终点 | 2026-11-17 23:59 UTC（`funding_rate` 到 2026-11-19 00:00），共 1,417 天 |
| 根目录 | `/share/binance_data/futures/um/{monthly,daily}/<dataset>/<SYMBOL>/...` |

10 个标的在 6 个数据集上的起止日期完全一致，覆盖区间内没有缺失的日文件。

## 样本切分

切点都在 UTC 00:00，区间为左闭右开。

| 段 | 区间 | 长度 | 用途 |
|---|---|---|---|
| 训练（IS） | `[2023-01-01, 2025-01-01)` | 2 年 | 研究、拟合、调参 |
| 验证 | `[2025-01-01, 2026-01-01)` | 1 年 | 选模型、定参数，可以反复看 |
| 测试（OOS） | `[2026-01-01, 2026-11-18)` | 约 10.5 个月 | 锁定，最终只跑一次 |

用法：直接传 `start` / `end`，例如 `bt.run(sig, d, start="2023-01-01", end="2025-01-01")`。`end` 不包含在内，不要写成 `"2024-12-31"`，否则会少最后一天。不传 `end` 时，默认一直评估到数据末尾，会包含测试段。

规则：

- **测试段只跑一次**。看过测试结果后再回头改策略，测试段就变成了样本内，需要等新数据重新留出测试段。
- **预热可以跨切点**。滚动算子可以使用切点之前的数据，例如测试段第一根 bar 的 `ts_zscore(…, 180)` 用 2025 年底的数据。只要这些数据不参与拟合就不算泄露。各段的绩效指标只从切点开始统计。
- **按信息可知的时刻归段**。资金费率 `2026-01-01 00:00` 那一条结算的是 `[2025-12-31 16:00, 2026-01-01 00:00)` 这一期，归验证段。其他数据也按对应时段归属，不按时间戳归属。

## 数据集一览

| 数据集 | 粒度 | 文件组成（2023 起） | 10 标的体积 | tsbt 已注册 |
|---|---|---|---|---|
| `klines` | 1m | monthly 2023-01 ~ 2025-11 + daily 2025-12-01 ~ 2026-11-17 | 1.3 GB | 是 |
| `metrics` | 5m | daily 2023-01-01 ~ 2026-11-17 | 0.4 GB | 是 |
| `funding_rate` | 每次结算 | 每个标的一个文件 `daily/funding_rate/<SYMBOL>.parquet` | < 2 MB | 否（目录结构不同） |
| `indexPriceKlines` | 1m | monthly 2023-01 ~ 2026-10 + daily 2026-11-01 ~ 11-17 | 1.1 GB | 否（加一条注册即可） |
| `premiumIndexKlines` | 1m | monthly 2023-01 ~ 2026-10 + daily 2026-11-01 ~ 11-17 | 1.0 GB | 否（加一条注册即可） |
| `aggTrades` | 逐笔（归集） | monthly 2023-01 ~ 2026-10 + daily 2026-11-01 ~ 11-17 | 141 GB | 是 |

1m 数据集都多一层 `1m/` 目录。当前读取规则是：某个月有 monthly 文件时，忽略该月所有 daily 文件。对 index / premium 来说，这条规则会丢掉月度文件里缺的那几天，见下文。

## 字段

时间列均为毫秒 epoch（UTC），`funding_rate` 除外。

- **klines / indexPriceKlines / premiumIndexKlines**：`open_time open high low close volume close_time quote_volume count taker_buy_volume taker_buy_quote_volume ignore`
  - `indexPriceKlines`：指数价格。`volume` 类字段恒为 0，`count` 通常是 60（每秒一次报价）。
  - `premiumIndexKlines`：溢价指数，就是计算资金费率用的 premium index，取值是比例（如 `-0.000555`）。`volume` 类字段恒为 0，`count` 通常是 12（每 5 秒一次）。
- **metrics**：`create_time timestamp sum_open_interest sum_open_interest_value count_toptrader_long_short_ratio sum_toptrader_long_short_ratio count_long_short_ratio sum_taker_long_short_vol_ratio`
- **funding_rate**：`timestamp(datetime[ns, UTC]) symbol funding_rate mark_price(字符串)`
- **aggTrades**：`agg_trade_id price quantity first_trade_id last_trade_id transact_time is_buyer_maker`

## 完整性检查

在 2023-01-01 ~ 2026-11-17 范围内逐行检查。

### klines：完整

10 个标的都是 2,040,480 根（= 1,417 × 1,440），没有缺口，没有重复，全部落在整分钟上。零成交分钟每个标的约 150 根（BTC 244 根），做 vwap 时需要退化为 twap，引擎已处理。

### aggTrades：完整

`agg_trade_id` 从起点到终点首尾相接，跨文件没有断档。唯一的瑕疵：`SOLUSDT-aggTrades-2025-07` 里 id `926014272` 出现两次（两笔成交不同：171.97 × 349、171.98 × 0.44），读取时不要按 id 去重。

### funding_rate：完整

每个标的 4,255 条，严格每 8 小时一次（00/08/16 UTC），没有缺失和重复。`mark_price` 在 2023-10-31 08:00 UTC 之前是空字符串。

> **时间戳是结算时刻，对应的是刚结束的上一期。** 例如 `00:00` 的 `funding_rate = 0.0001`，表示 `[前一天 16:00, 00:00)` 这一期按 0.0001 结算了资金费，不代表 00:00 之后要收多少。
>
> - 作为特征：这条记录要到 `00:00` 才可知，只能给收盘时刻 ≥ 00:00 的 bar 使用。
> - 计算持仓资金费：`00:00` 这条对应 16:00 ~ 00:00 持有的仓位，不对应 00:00 之后的仓位。
>
> 已用 `premiumIndexKlines` 反算验证：按 `F = P + clamp(I − P, ±0.05%)` 计算，`P` 取窗口内 1m 溢价指数的线性加权均值。用时间戳**之前** 8h 的窗口，10 个标的与实际值的相关系数为 0.986 ~ 0.999，平均绝对误差 3e-6 ~ 8e-6。用**之后** 8h 的窗口，相关系数只有 0.64 ~ 0.81，误差大一个数量级。

**截断**：资金费率不是溢价的线性函数。

- 利率项 `I` 一般是 0.01%，**BNBUSDT 是 0**。
- 只要 `P` 落在 `I ± 0.05%` 带内，资金费率就被截成 `I`。2023 年以来，有 26% ~ 45% 的期恰好等于 0.0001；BNB 有 55% 的期恰好等于 0。这个比例与 `P` 落在带内的比例一致。
- 数值精度为小数点后 8 位。

因此资金费率的分布在 `I` 处有一个很大的质量点。做特征时不要假设它连续，例如做 z-score 时要注意。

反算与实际值仍有差异，主要有两个原因：一是用的是 1m 溢价指数，Binance 实际按每 5 秒采样计算；二是极端行情（如 2025-10-11 00:00）。另外 2023-08-29 16:00 这一期在多个标的上偏差都很大，原因未查。

### indexPriceKlines / premiumIndexKlines：月度文件缺天，daily 文件可补齐

Binance 官方的**月度 zip 本身就缺下面这几天**（已从 data.binance.vision 下载原始 zip 核对行数），本地 monthly parquet 与原始 zip 一致。但这几天的 **daily 文件在官方和本地都有，且完整**（每个 1,440 行）。

问题出在读取规则：当前规则是某个月有 monthly 文件时，忽略该月所有 daily 文件，这几天因此被丢掉。改成“monthly 缺的日期用 daily 补”之后，这些整天缺口全部消失。

| 数据集 | 日期 | 标的 | monthly | daily |
|---|---|---|---|---|
| index | 2023-02-13 | BTC BNB AVAX | 缺整天 | 完整 |
| index | 2023-02-24 | 全部 10 个 | 缺整天 | 完整 |
| index | 2023-04-07、04-08 | BTC BNB AVAX | 缺整天 | 完整 |
| index | 2026-06-29 | 全部 10 个 | 缺整天 | 完整 |
| premium | 2023-02-24 | BTC BNB AVAX LINK | 缺整天 | 完整 |
| premium | 2023-04-09 | ETH SOL XRP DOGE ADA LTC | 缺整天 | 完整 |
| premium | 2026-06-29 | 全部 10 个 | 缺整天 | 完整 |

补齐之后，剩下的是**源数据本身的缺失**（官方 daily zip 也是这个行数）：

| 数据集 | 日期 | 缺失 | 标的 |
|---|---|---|---|
| index | 2024-08-12 | 2 分钟（1,438 / 1,440） | 全部 10 个 |
| premium | 2023-11-10 | 14 分钟（1,426 / 1,440） | 全部 10 个 |
| premium | 2024-08-12 | 2 分钟（1,438 / 1,440） | 全部 10 个 |

### metrics：基本完整，时间戳需要对齐

- 每个标的缺 134 ~ 193 个 5m 时点（约 0.04%）。最大的一处在 **2024-02-16**，10 个标的都只有 163 / 288 个点，官方 daily zip 也只有 163 行，属于源数据缺失；其余是零星缺失（如 2023-09-12、2025-08-29 各缺 3 个）。
- 时间戳有偏移：每个标的有约 2,000 ~ 5,500 行不在 5m 整点上，最多晚 35 秒（BTC 2,987 行）。使用时需要向下取整到 5m。
- 日文件交界处各有 1 个重复时间戳（00:00），如 BTC 2025-01-24 00:00。
- 文件内的行没有按时间排序。
- `count_toptrader_long_short_ratio` 每个标的约 80 个空值，`sum_taker_long_short_vol_ratio` 有 0 ~ 10 个空值；`sum_open_interest` 没有空值。
