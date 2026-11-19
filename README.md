# tsbt

Binance USDⓈ-M 永续合约的**时序信号向量化回测框架**。信号是连续仓位（如 `0.2 / -0.12`），bar 由 1 分钟数据合成并缓存，引擎是纯 numpy 闭式计算，没有沿时间的循环。设计细节见 [SPEC.md](SPEC.md)。

```bash
uv sync                 # Python 3.12，依赖见 pyproject.toml
uv run pytest           # 场景测试（约 15 s）
```

## 快速开始

```python
import numpy as np
import polars as pl
import tsbt as bt
from tsbt import ops as op

d = bt.load("BTCUSDT", freq="4h", fields=["close", "volume", "oi"])   # 首次会建缓存
sig = op.tanh(op.ts_zscore(op.ts_returns(d.close, 6), 180)) * 0.5

res = bt.run(sig, d, price="vwap", fee=2e-4)     # 复利、延迟 1 根 bar 成交
print(res)                  # apr / sharpe / mdd / calmar / turnover
res.summary()               # polars DataFrame，含 start end days 及附加指标
res.yearly(); res.monthly(); res.daily()
res.plot()                  # 5 张图（xy）；notebook 里逐张显示，res.plot("nav") 只画一张
res.save("runs/btc_mom_4h") # metrics.json config.json series.parquet yearly.parquet *.png
```

## 时间与成交规则

- 全部 UTC。bar 以开盘时间标记，覆盖 `[open, open+F)`，对齐 UTC 00:00（可用 `offset=` 分钟平移相位）。频率可取任意能整除 1440 分钟的值：`15m 1h 2h 4h 6h 12h 1D`……
- `signal[t]` 视为“截至 bar t 收盘可知”。**引擎统一做位移**：在 bar `t+delay` 内部按 `price` 成交，`delay` 默认 1，`delay < 1` 直接报错。**不要自己 shift 信号。**
- `price`：`open`（T 的第一根 1m open）、`close`（T 的最后一根 1m close）、`twap`（T 内各分钟 VWAP 的等权均值）、`vwap`（T 内 Σ成交额/Σ成交量，零成交时退化为 twap）。`exec_window=N` 只用 bar 开头 N 分钟算 twap/vwap。
- 净值按 bar 收盘价盯市；结果的时间戳是 bar 收盘时刻。
- 结果从“第一根有可用信号的 bar”开始（这根 bar 开盘时净值 = 1）。信号里的 NaN 和 ±inf 都表示“无信号”，默认平仓（`nan_policy="flat"`），可改为沿用上一目标（`"hold"`）。

## 费用与换手

- 单一费率 `fee`（手续费 + 滑点，线性），默认万 2，按成交名义收取。
- **每根 bar 都调回目标仓位**。即使信号连续两期都是 0.2，价格变动后实际仓位已漂移，调回 0.2 的成交量也计入换手和费用。
  - 复利：漂移后权重 `w⁻ = θ(1+R)/(1+Σθ·R)`。例：0.2 → 0.2，价格 +5%，换手 0.007921。
  - 单利：漂移后名义 `θ(1+R)`。同一例子换手 0.01。
- `turnover` 指标是日均双边换手（从 0 开到 0.2 记 0.2）。

## 资金口径

| `compound=` | 仓位含义 | 净值 | APR | MDD |
|---|---|---|---|---|
| `True`（默认） | 名义 / 当前权益 | 连乘 | 年化复合 | 相对回撤（负数） |
| `False` | 名义 / 固定本金 1 | 累加 | 算术年化 | 以本金为单位（负数） |

复利模式下权益 ≤ 0 时回测终止，结果标记 `bankrupt=True`。

## 指标

`apr sharpe calmar turnover mdd start end days`，另附 `total_return apr_gross sharpe_gross fee_drag avg_abs_pos long_ratio bench_apr bench_sharpe bench_mdd bankrupt`。

- Sharpe 统一用 **UTC 日频**净值收益 × √365，所以不同 bar 频率的结果可以直接比较。首尾不足一天的部分各算一个观测。MDD 用 bar 级净值。
- 基准 = 同一标的、信号恒为 1、同一资金口径、费率 0（单标的复利即买入持有）。组合的基准是**买入不动**的篮子：每个标的在起点（晚上市的在上市时）按权重买入一次，之后不调仓，等待上市的份额是现金。它不依赖策略表现：静态分配时初始权重与分配相同，动态分配（滚动 Sharpe、反波动、时变数组、自定义函数）时用等权 `1/N`。

## 多标的

```python
syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
d = bt.load(syms, freq="1h")                       # 字段为 (T, N)，并集时间轴，未上市为 NaN
sig = op.tanh(op.ts_zscore(op.ts_returns(d.close, 24), 720)) * 0.5

pr = bt.run(sig, d)                                # 默认 mode="portfolio"：共享资金池，等权 1/N
pr.summary()      # 组合 1 行 + 每个标的的独立回测各 1 行
pr.contrib()      # 各标的的盈亏 / 换手贡献
pr.components["ETHUSDT"].plot("nav")

br = bt.run(sig, d, mode="independent")            # 每列一个独立账户
bt.run(np.column_stack([s1, s2, s3]), bt.load("BTCUSDT", "4h"),
       names=["fast", "mid", "slow"])              # 单标的 K 组参数批量
```

资金分配 `alloc=`（行 t 视为 bar t 收盘可知，和信号一样由引擎延迟）：

| 写法 | 含义 |
|---|---|
| `None` / `bt.alloc.equal()` | 静态 1/N（未上市标的的份额闲置） |
| `bt.alloc.equal_active()` | 当期已上市标的等权 |
| `bt.alloc.fixed([...])` 或 `(N,)` 数组 | 静态权重 |
| `(T, N)` 数组 | 任意时变权重，原样使用 |
| `bt.alloc.inv_vol(window)` | 反比于各标的独立策略收益的滚动波动 |
| `bt.alloc.rolling_sharpe(window, floor=0.0)` | 正比于各标的独立策略的滚动 Sharpe，低于 floor 截断；全部截断时空仓 |
| 可调用 `f(ctx) -> (T, N)` / `bt.alloc.custom(f)` | 完全自定义；`ctx` 有 `ts symbols signal data listed ret ret_gross` |

通用选项：`normalize=True`（每行归一到和为 1）、`rebalance_every=k`（每 k 根 bar 更新一次权重）。动态分配器在历史不足时用已上市标的等权。

标的数据提前结束（下市或数据截止）时，在最后一根 bar 按执行价平仓并计费。区间中部整根 bar 没有数据时不成交，沿用上一期目标。

## 数据与缓存

- 数据根默认 `/share/binance_data/futures/um`（`TSBT_DATA_ROOT`），缓存默认在项目根目录的 `.cache/`（已 gitignore；`TSBT_CACHE_DIR` 可覆盖，非源码安装时退回 `~/.cache/tsbt`），也可用 `bt.set_paths(...)`。
- 某个月有 monthly 文件时，该月的 daily 文件忽略。
- **缺数据会提示**：由 1 分钟数据合成 bar 时，如果标的上市区间内的源数据有缺口，发出 `bt.MissingDataWarning`，写明标的、缺多少和起止时间。klines 字段按分钟检查（一根 bar 只缺几分钟也会报），每个标的报一次；metrics 和特征字段是稀疏的，整根 bar 没有任何值才报，每个字段报一次。提示只在合成时出现一次，之后从缓存加载不再提示；合成覆盖标的的全部历史，不受 `start` / `end` 影响。
- 第一次使用某个（标的 × 数据集）时，把源文件合成为 1 分钟数组存入 zarr；之后每次加载只比对源文件清单（文件名 / 大小 / mtime），新文件只增量处理。bar 由分钟数据 reshape 归约而来，同样缓存。
- 内置字段：`open high low close volume quote_volume count taker_buy_volume taker_buy_quote_volume vwap twap`（klines）；`oi oi_value top_ls_acct top_ls_pos ls_acct taker_ls_vol`（metrics，5 分钟一条，bar 取最后一条；记录时间恰在 bar 边界时归入下一根 bar）。

```bash
uv run tsbt cache build --symbols BTCUSDT ETHUSDT --freq 1h 4h   # 预热（全部标的用 --all）
uv run tsbt cache status
uv run tsbt cache clear --symbol ETHUSDT
```

## bar 内聚合

`d.m1` 是与 bar 严格对齐的分钟视图（长度 `T*k`），先在分钟级算中间量，再用 `d.agg` 聚合：

```python
m = d.m1
ret = op.ts_returns(m.close, 1)
rv  = d.agg(ret, "std")                      # bar 内已实现波动
pvc = d.agg((ret, m.volume), "corr")         # 双输入：corr cov beta wmean
q90 = d.agg(m.close, "quantile", q=0.9)
x   = d.agg(m.volume, "sum", window=30)      # 只用 bar 开头 30 分钟
y   = d.agg(m.high - m.low, lambda b: np.nanmax(b, axis=1))   # 任意可调用，b 形状 (T, k[, N])
d.save_field("rv", rv)                        # 缓存；之后 bt.load(..., fields=["rv"])
                                              # overwrite=True 整体替换，范围外的旧值丢弃
```

内置归约：`sum mean std var first last max min median count skew kurt quantile`。

## 自定义分钟特征（aggTrades 等）

框架不预置 aggTrades 特征。注册一个特征集后，它的输出就和内置字段一样用：`bt.load(sym, "1h", fields=["big_buy_amt"])`。

```python
# ① 聚合表达式：框架包好 GROUP BY 分钟
bt.feature_set("flow", source="aggTrades",
    sql={"big_buy_amt": "sum(price*quantity) FILTER (WHERE NOT is_buyer_maker AND price*quantity >= 1e5)"},
    bar_agg={"big_buy_amt": "sum"}, fill={"big_buy_amt": 0.0})

# ② 完整 DuckDB 查询：{src} 是当前源文件，必须返回 ts 列（分钟起点，ms）
bt.feature_set("micro", source="aggTrades", query="""
    SELECT (transact_time // 60000) * 60000 AS ts, stddev_samp(price) AS px_std
    FROM {src} GROUP BY 1""")

# ③ Python 函数：拿到单个源文件，返回 polars DataFrame（ts + outputs）
@bt.feature_set("custom", source="aggTrades", outputs=["sell_qty"], bar_agg={"sell_qty": "sum"}, version=1)
def custom(chunk: bt.SourceChunk):
    return (chunk.scan().filter(pl.col("is_buyer_maker"))
            .group_by(ts=(pl.col("transact_time") // 60000) * 60000)
            .agg(sell_qty=pl.col("quantity").sum()).collect())
```

- 输出每分钟至多一行，`ts` 是 60000 的整数倍且落在该文件的时间范围内，否则报错。
- 每个源文件（一个月或一天）单独计算、单独缓存，跨文件的状态不可见。出错或 Ctrl-C 时已算完的文件会先落盘，重跑只补未完成的文件。
- 定义改变（SQL 文本 / 函数源码 + `version`）只重算该特征集。
- 并发文件数 × 每个文件的 DuckDB 线程数（`threads=`，默认 4）不会超过机器核数。
- `bar_agg` / `fill` 的键必须是输出列名，`bar_agg` 的值必须是内置归约名（`sum mean last max ...`），拼错会直接报错。

> **⚠ Python 特征的缓存失效盲区**：框架只看**被装饰函数自身的源码**和 `version`。如果函数调用了别处的辅助函数，或用到外部定义的阈值（如 `BIG = 1e5`），改了它们框架看不出来，会**静默沿用旧缓存**，回测照常跑完、不报错。**改动这类外部逻辑后，把 `version` 加 1**，或者把阈值写进函数内部。`sql=` / `query=` 特征不受影响（整段 SQL 文本参与判断）。

## 算子 `tsbt.ops`

沿用 AlphaForge 的时序算子语义（窗口 `d` 的单位是 bar）：窗口内有效样本少于 `min(d, max(2, ⌈d/2⌉))` 时输出 NaN，NaN 和 ±inf 样本跳过，std/cov 用 ddof=1，非有限结果一律转 NaN。窗口可以比样本长；常数窗口的 `ts_zscore / ts_ir / ts_corr` 为 NaN，`ts_covariance` 为 0。各列独立，`(T,)` 与 `(T, N)` 通用。

- 逐元素：`add subtract multiply divide inverse reverse abs sign log exp sqrt power signed_power tanh sigmoid clamp max min round round_down floor ceiling fraction if_else greater greater_equal less less_equal equal not_equal and_ or_ not_ is_nan is_not_nan is_finite nan_mask nan_out to_nan purify left_tail right_tail`
- 位移 / 填充：`ts_delay ts_delta ts_returns log_diff ts_backfill`
- 滚动：`ts_sum ts_mean ts_std_dev ts_min ts_max ts_median ts_rank ts_zscore ts_scale ts_av_diff ts_ir ts_arg_max ts_arg_min ts_product ts_skewness ts_kurtosis ts_moment ts_quantile ts_entropy ts_count_nans inst_tvr ts_decay_linear ts_decay_exp_window filter`
- 双序列：`ts_corr ts_covariance ts_regression ts_seq_beta`
- 递推：`ts_ema ts_sma_cn hump hump_decay jump_decay trade_when keep`

与 AlphaForge 唯一的语义差异：`hump(x, threshold)` 用绝对阈值（AlphaForge 按截面 |x| 均值缩放）。截面算子未移植。

外部算好的信号可直接传给 `bt.run`；带“可知时刻”的外部序列用 `d.align(ts, values, ffill=False)` 对齐到 bar（可知时刻落在 `(open, close]` 的值归入该 bar）。

## 图表

`res.plot()` / `res.save()` 生成 5 张 PNG（用 [xy](https://github.com/reflex-dev/xy) 绘制）：

| 文件 | 内容 |
|---|---|
| `nav.png` | 费后净值、费前净值、标的基准；下方是费后回撤 |
| `position.png` | 标的价格（组合为基准净值）、仓位（组合为净 / 总敞口）、每日换手 |
| `long_short.png` | 多头、空头、合计的累计盈亏（费前） |
| `periodic.png` | 月度收益热力图 + 年度收益柱（标注当年 Sharpe / MDD） |
| `rolling_sharpe.png` | 90 天滚动 Sharpe，附基准 |
