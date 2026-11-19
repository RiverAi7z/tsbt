# tsbt — 时序信号向量化回测框架 SPEC

> 状态：v0.1 已实现（2026-11-18）。实现中做出的细化见第 17 节。
> 范围：Binance USDⓈ-M 永续合约，时序（time-series）仓位信号回测

---

## 1. 目标与非目标

### 1.1 目标

1. 对**连续仓位信号**（如 `0.2 / 0.3 / -0.12`，而非 ±1）做向量化回测。
2. 支持多种频率：`1h / 2h / 4h / 6h / 12h / 1D`（以及任何能整除 1440 分钟的周期）。
3. 执行价可选 `open / twap / vwap / close`；**信号由 T-1 及之前的数据算出，只能在 T 这根 bar 内成交**，结构上杜绝使用 T-1 的价格成交。
4. 所有 bar 数据由 **1 分钟基础数据**合成；首次运行落盘缓存，之后不再重复合成。
5. 因子输入来自 `klines / metrics / aggTrades`，算子体系沿用 AlphaForge 的时序子集。
6. 费用模型：单一固定费率（手续费 + 滑点合并，线性），默认万 2，不区分 maker/taker。
7. **漂移换手**：连续两期信号同为 0.2，价格变化导致实际仓位偏离，调回目标产生的换手与费用必须计入。
8. 输出指标 `apr / sharpe / calmar / turnover / mdd / start / end / days`，以及 5 张图。
9. 支持单标的、多标的批量、多标的共享资金池组合（资金分配规则高度可定制）。

### 1.2 非目标（第一版不做）

| 不做 | 说明 |
|---|---|
| 资金费率 | 已确认不考虑。策略与基准都只含价格盈亏和手续费。 |
| 调仓阈值 / 不交易带 | 已确认不需要。每根 bar 严格调回目标仓位；降换手在信号侧用 `hump_decay / trade_when / keep` 等算子处理。 |
| 字符串表达式 DSL | 已确认只做 Python 函数式算子 + 直接传数组。 |
| 截面算子 | `rank / zscore / group_* / fund_* / vec_*` 等截面、分组、基本面算子不移植。数组是 `(T, N)`，后续要加不受结构限制。 |
| 保证金 / 强平 / 最小下单量 / 盘口冲击 | 线性成本假设下不建模。复利模式下权益 ≤ 0 时回测终止并标记 `bankrupt`。 |
| HTML 报告 | 已确认不需要，只出 PNG。 |

---

## 2. 采访确认的决策

| # | 问题 | 结论 |
|---|---|---|
| 1 | 信号的资金口径 | **复利、单利都支持**，`compound=True/False`，默认复利 |
| 2 | twap/vwap 成交窗口 | **可配置，默认整根 T bar**；可设为 bar 开头 N 分钟 |
| 3 | 资金费率 | **不考虑** |
| 4 | 信号书写方式 | **Python 函数式算子**，也可直接传入外部算好的数组 |
| 5 | 标的范围 | **单标的** 与 **多标的 + 组合汇总** 都支持 |
| 6 | bar 内聚合 | 固定规则 + **自定义 bar 内聚合**（分钟级中间量 → bar） |
| 7 | aggTrades 特征 | **不预置任何特征**，提供自由度最大的用户自定义机制 |
| 8 | Sharpe 口径 | **统一用日频净值**（UTC 日），√365 年化 |
| 9 | 组合资金合并 | **共享资金池**；分配权重自由度要大（如按 Sharpe 分配），默认等权 |
| 10 | 调仓阈值 | **不需要**，引擎保持纯闭式向量化 |
| 11 | 图表 | 净值+回撤、仓位与换手、多空拆分、月度/年度收益、滚动 Sharpe |
| 12 | 图表库 | **`xy`**（reflex-dev/xy），只出 PNG，不要 HTML |

第 15 节列出了采访没覆盖、由我先定下的默认口径，请过目。

---

## 3. 数据现状（2026-11-18 实测）

根目录 `/share/binance_data/futures/um/{daily,monthly}/<dataset>/<SYMBOL>/...parquet`

| 数据集 | 粒度 | 标的数 | BTCUSDT 覆盖 | 备注 |
|---|---|---|---|---|
| `klines` | 1m | 690 | monthly 2020-01 ~ 2025-11，daily 2025-12-01 ~ 2026-09-22 | 多一层 `1m/` 目录。BTC/ETH 共 3,538,080 根，**无缺口、无重复**；有少量零成交分钟（BTC 369 根） |
| `metrics` | 5m | 419 | daily 2021-12-01 ~ 2026-09-30 | 文件内**行未按时间排序**；BTC 有 2,987 行时间戳不在 5m 整点上、1 个重复时间戳 |
| `aggTrades` | 逐笔 | 48 | monthly 2023-12 ~ 2026-09，daily 2026-10-01 ~ 10-17 | BTC 单月约 667 MB / 3,600 万行；monthly 合计 130 GB。**daily 与 monthly 存在重叠**（如 ETHUSDT daily 2025-09 ~ 11 已被 monthly 覆盖） |

列结构：

- `klines`：`open_time(ms) open high low close volume close_time quote_volume count taker_buy_volume taker_buy_quote_volume`
- `metrics`：`create_time timestamp(ms) sum_open_interest sum_open_interest_value count_toptrader_long_short_ratio sum_toptrader_long_short_ratio count_long_short_ratio sum_taker_long_short_vol_ratio`
- `aggTrades`：`agg_trade_id price quantity first_trade_id last_trade_id transact_time(ms) is_buyer_maker`

三个数据集都有的标的只有 37 个；`klines` 是唯一全覆盖的数据集，回测价格只依赖它。

---

## 4. 架构总览

```
原始 parquet (只读)
   │  klines ── polars ──┐
   │  metrics ─ polars ──┤
   │  aggTrades ─ duckdb ┤  ← 用户自定义特征 (SQL / Python)
   ▼                     ▼
 L1 分钟库  m1/<symbol>/<field>      zarr v3，规则 1m 网格，缺失 = NaN
   │  reshape (T, k) + nan-reduce（纯 numpy）
   ▼
 L2 bar 库  bar/<freq>/<symbol>/<field>   zarr v3
   │
   ▼
 BarData (T, N) numpy ──► ops（算子）──► signal (T, N)
                                           │
                                           ▼
                      engine（纯 numpy 闭式，无时间循环）
                                           │
                             Result ──► metrics / xy 图 / 落盘
```

### 技术选型

| 组件 | 选择 | 理由 |
|---|---|---|
| 回测引擎 | **numpy** | 每期调回目标仓位时漂移权重有闭式解，`cumprod / cumsum` 即可，沿时间轴无 Python 循环 |
| 缓存存储 | **zarr v3** | 1m 是规则网格，时间索引隐式（下标 = (t − start)/60s），读出来直接是 ndarray；每个字段一个数组，新增用户特征 = 新增一个数组，不必重写已有数据；支持尾部追加与分块读取。与 AlphaForge 存储后端一致 |
| klines / metrics 读取 | **polars** | 懒扫描 + 列裁剪，去重、排序、对齐网格 |
| aggTrades 聚合 | **duckdb** | 单文件数亿行的 `GROUP BY` 可流式执行、不爆内存；用户直接写 SQL，自由度最大 |
| 滚动算子 | **bottleneck + numba** | 与 AlphaForge 相同：滚动统计走 bottleneck，递推类算子（EMA、hump、trade_when）走 numba 核。引擎本身不依赖 numba |
| 图表 | **xy ≥ 0.0.9** | 指定库。用其 `xy.pyplot` 原生模式（已冒烟验证，见 §10） |
| 构建 | **uv**，Python 3.12 | xy 要求 ≥ 3.11 |

---

## 5. 时间约定与防未来函数

1. 全部时间为 **UTC**，毫秒整数存储，对外为 `datetime64[ms]`。
2. 1m 行以 `open_time = m` 标记，覆盖 `[m, m+60s)`，其内容在 `m+60s` 才完全可知。
3. 频率 `F`（分钟数 `k`，须整除 1440）的 bar 以**开盘时间**标记，覆盖 `[ts, ts+F)`，对齐 UTC 00:00；可选 `offset`（分钟，< k）整体平移相位。
4. **行对齐规则**：任何 bar 级数组的第 `t` 行，只包含 `≤ bar t 收盘时刻`的信息。所有内置算子都是因果的（只看过去窗口），所以用户算出的 `signal[t]` 天然满足“截至 bar t 收盘可知”。
5. **引擎统一做位移**：`target[t+1] = signal[t]`。用户**不需要也不应该**自己 shift。参数 `delay`（默认 1）可加大延迟，`delay < 1` 直接报错——这是“不允许用 T-1 价格成交”的硬约束。
6. 成交发生在 bar T **内部**：`open` = T 的第一根 1m 的 open；`close` = T 的最后一根 1m 的 close；`twap / vwap` = T 内成交窗口的均价。成交窗口永远是 T 的子区间。
7. `metrics` 记录时间戳为 `τ` 时，归入 `open_time = floor(τ, 1m)` 的分钟行（视为 τ 之后才可知）。因此恰好盖在 bar 边界上的记录（如 03:00:00）属于**下一根** bar，不会泄漏给在 03:00 收盘的 bar。

---

## 6. 数据层

### 6.1 数据集注册

数据集是一条注册记录：`名称 / 目录模式 / 时间列 / 粒度`。已注册 `klines`、`metrics`、`aggTrades`、`trades`（字段均与真实文件核对过）。`liquidationSnapshot`、`indexPriceKlines`、`premiumIndexKlines` 目录结构相同，加一条注册即可被自定义特征使用。

**monthly / daily 取舍**：某个月存在 monthly 文件时，该月全部以 monthly 为准；daily 文件只用于没有 monthly 覆盖的日期。合并后按时间键去重（保留最后一条）并排序。

### 6.2 L1 分钟库

- 路径：`$TSBT_CACHE_DIR/store.zarr/m1/<SYMBOL>/<field>`，1-D `float64`，zstd 压缩，约 50 万点（≈ 1 年）一个 chunk。
- 所有数组共用一个全局起点（2019-01-01 UTC，下标 = 距起点的分钟数），读取任意时间段只是一次切片，多标的对齐不需要各自的偏移。
- 组属性：每个构建器一份清单 `build:<key>`，记录定义哈希、字段、已构建的源文件（文件名 → 大小 / mtime）、数据起止时间和版本号。
- 网格内缺失的分钟填 `NaN`。
- **按需、按（标的 × 数据集）懒构建**：`load` 时发现缺失才构建；也可用 CLI 预热。

### 6.3 字段目录

| 字段 | 来源 | 分钟级 | bar 聚合规则 |
|---|---|---|---|
| `open` | klines | 原值 | first |
| `high` | klines | 原值 | max |
| `low` | klines | 原值 | min |
| `close` | klines | 原值 | last |
| `volume` | klines | 原值 | sum |
| `quote_volume` | klines | 原值 | sum |
| `count` | klines | 原值 | sum |
| `taker_buy_volume` | klines | 原值 | sum |
| `taker_buy_quote_volume` | klines | 原值 | sum |
| `vwap` | klines | `quote_volume / volume`，零成交分钟取 `close` | `Σquote_volume / Σvolume`，整根零成交取 `twap` |
| `twap` | klines | 同分钟级 `vwap` | 分钟级 `vwap` 的等权平均 |
| `oi` | metrics `sum_open_interest` | 稀疏（5m 一条） | last（bar 内最后一条有效值） |
| `oi_value` | metrics `sum_open_interest_value` | 稀疏 | last |
| `top_ls_acct` | metrics `count_toptrader_long_short_ratio` | 稀疏 | last |
| `top_ls_pos` | metrics `sum_toptrader_long_short_ratio` | 稀疏 | last |
| `ls_acct` | metrics `count_long_short_ratio` | 稀疏 | last |
| `taker_ls_vol` | metrics `sum_taker_long_short_vol_ratio` | 稀疏 | last |
| 用户特征 | 任意数据集 | 用户定义 | 用户声明（默认 last） |

`load_m1` 读取 metrics 字段时默认前向填充，上限 15 分钟；bar 级不填充。

### 6.4 L2 bar 库与合成方法

- 路径：`store.zarr/bar/<freq>[+<offset>]/<SYMBOL>/<field>`。
- **合成方法**：1m 网格规则且无洞（缺失为 NaN），所以取整根 bar 对齐的区间后 `x.reshape(T, k)`，沿第 1 轴做 nan-aware 归约即可。固定规则字段与自定义聚合走同一条路径，无 group-by。
- **完整性**：只有 k 根分钟全部落在该标的 `[首根分钟, 末根分钟]` 范围内的 bar 才生成；首尾不完整的 bar 丢弃。范围内部 kline 缺失的分钟按 NaN 参与归约；整根 bar 无有效分钟则该 bar 为 NaN。
- L2 记录其依赖的 L1 版本号；L1 变化后 L2 自动重建（纯内存 reshape，秒级以内）。

### 6.5 执行价

| `price` | 定义 | 受 `exec_window` 影响 |
|---|---|---|
| `open` | bar T 第一根 1m 的 open | 否 |
| `close` | bar T 最后一根 1m 的 close | 否 |
| `twap` | 窗口内分钟级 `vwap` 的等权平均 | 是 |
| `vwap` | 窗口内 `Σquote_volume / Σvolume`；窗口零成交时退化为 `twap` | 是 |

`exec_window=None`（默认）= 整根 bar，直接用 L2 的 `twap / vwap` 字段。`exec_window=N`（分钟，`1 ≤ N ≤ k`）= bar 开头 N 分钟，取 reshape 后的前 N 列归约，结果以 `twap@N / vwap@N` 缓存进 L2。

### 6.6 自定义 bar 内聚合

`BarData` 同时持有 bar 级字段和与之**严格对齐**的分钟级视图（长度恰为 `T × k`）。用户先在分钟级算中间量，再聚合到 bar：

```python
d = bt.load("BTCUSDT", freq="4h")           # d.close 等: (T,)
m = d.m1                                    # 分钟视图，m.close: (T*240,)

ret = op.ts_returns(m.close, 1)             # 分钟级中间量
rv  = d.agg(ret, "std")                     # bar 内已实现波动      -> (T,)
pvc = d.agg((ret, m.volume), "corr")        # bar 内量价相关        -> (T,)
up  = d.agg(m.volume * (ret > 0), "sum") / d.volume
rng = d.agg(m.high - m.low, lambda b: np.nanmax(b, axis=1))   # 任意可调用
```

- `how` 内置：`sum mean std var first last max min median count skew kurt quantile(q)`，双输入 `corr cov beta wmean`；或任意可调用 `f(block) -> array`，`block` 形状 `(T, k[, N])`，沿第 1 轴归约。
- `d.agg(x, how, window=N)`：只聚合 bar 开头 N 分钟。
- 结果行 `t` 只用到 bar t 内的分钟，符合 §5 的对齐规则。
- **缓存**：`d.save_field("rv", rv)` 把结果按（标的 × 频率）存入 L2 的 `user/` 命名空间，之后 `fields=["rv"]` 即可加载；覆盖需显式 `overwrite=True`（框架无法得知上游计算是否变化，失效由用户负责）。

> 采访预览里写的是 `bt.bar_agg(ret, '4h', 'std')`。裸 ndarray 不带时间轴，无法保证与 bar 对齐，所以把聚合挂在数据对象上（`d.agg`），对齐由构造保证。独立的 `bt.load_m1(...)` 仍然提供，用于纯分钟级研究。

### 6.7 用户自定义分钟特征（aggTrades 等）

框架不预置任何 aggTrades 特征。用户注册**特征集**（一次扫描源文件产出多个 1m 特征），框架负责逐文件执行、并行、落盘、增量与失效。三种写法，自由度递增：

```python
# ① 聚合表达式：框架包好 GROUP BY 分钟；每个表达式一个特征
bt.feature_set(
    "flow", source="aggTrades",
    sql={
        "big_buy_amt":  "sum(price*quantity) FILTER (WHERE NOT is_buyer_maker AND price*quantity >= 1e5)",
        "big_sell_amt": "sum(price*quantity) FILTER (WHERE is_buyer_maker AND price*quantity >= 1e5)",
        "max_trade":    "max(price*quantity)",
    },
    bar_agg={"big_buy_amt": "sum", "big_sell_amt": "sum", "max_trade": "max"},
    fill={"big_buy_amt": 0.0, "big_sell_amt": 0.0},   # 无成交分钟的取值，默认 NaN
)

# ② 完整查询：自己写 CTE / 窗口函数，{src} 是当前源文件，须返回 ts 列（分钟起点，ms）
bt.feature_set("micro", source="aggTrades", query="""
    WITH t AS (SELECT *, price - lag(price) OVER (ORDER BY agg_trade_id) AS dp FROM {src})
    SELECT (transact_time // 60000) * 60000 AS ts,
           count(*) FILTER (WHERE dp <> 0) AS n_price_moves,
           stddev_samp(price)              AS px_std
    FROM t GROUP BY 1""")

# ③ Python 可调用：拿到单个源文件，返回 polars DataFrame（ts + 特征列）
@bt.feature_set("custom", source="aggTrades", version=1, bar_agg={"x": "mean"})
def custom(chunk: bt.SourceChunk) -> pl.DataFrame:
    lf = chunk.scan()            # polars LazyFrame；也可用 chunk.sql(...) 或 chunk.path
    ...
```

约定与限制：

- 输出必须是**每分钟至多一行**，`ts` 为 60000 的整数倍且落在该源文件的时间范围内，特征列为数值；否则报错。
- 特征名全局唯一，注册后与内置字段同等使用：`bt.load(sym, freq="1h", fields=["close", "big_buy_amt"])`。
- 定义写在用户自己的模块里，`import` 即注册；框架不持久化代码。
- **失效**：缓存记录定义哈希（SQL 文本，或函数源码 + `version`）。定义改变只重算该特征集，不影响其他字段。
- **⚠ Python 特征的失效盲区**：哈希只覆盖被装饰函数**自身的源码**和 `version`。函数调用的辅助函数、读取的外部常量（如 `BIG = 1e5`）改了，框架看不出来，会**静默沿用旧缓存**，回测照常跑完不报错。改动这类外部逻辑后须把 `version` 加 1，或把阈值写进函数内部。`sql=` / `query=` 特征不受影响（整段 SQL 文本参与哈希）。
- **进度可续**：按源文件记录完成清单，中断后续跑；出现新文件时只处理新文件。
- **文件相互独立**：每个源文件（一个月或一天）单独计算，跨文件边界的状态（如上个月最后一笔成交）不可见。
- 标的没有该数据集时，加载报错并指明缺哪个数据集。
- 成本预期：aggTrades 以扫描为主，BTC 单月 667 MB；按文件并行（`workers` 可配）。

### 6.8 缓存一致性

- **首次**：构建并落盘。**之后**：只比对源文件清单（文件名 / 大小 / mtime），不读源数据。
- 尾部新增文件 → 增量追加；已有文件发生变化 → 该（标的 × 数据集）重建。
- 构建期间按标的加文件锁，避免并发写。
- 缓存位置 `TSBT_CACHE_DIR`（默认项目根目录的 `.cache/`，非源码安装时为 `~/.cache/tsbt`），数据根 `TSBT_DATA_ROOT`（默认 `/share/binance_data/futures/um`）。
- CLI：`tsbt cache build (--symbols ... | --all) --freq ... [--workers N]`（必须指明标的，全量构建要显式写 `--all`）、`tsbt cache status`、`tsbt cache clear [--symbol ...] [--field ...]`。

### 6.9 加载 API

```python
d = bt.load(symbols, freq="4h", fields=None, start=None, end=None, offset=0)
m = bt.load_m1(symbols, fields, start=None, end=None)
bt.symbols(dataset="klines")      # 可用标的
bt.fields()                       # 内置字段 + 已注册用户特征
```

- `symbols` 传字符串 → 字段为 `(T,)`；传列表 → `(T, N)`，列顺序同列表。
- 多标的统一到**并集时间轴**，未上市 / 已下市区间为 NaN。
- `start / end` 自动对齐到 bar 边界；`fields=None` 加载 klines 全部内置字段。
- `d.ts`（bar 开盘时间）、`d.symbols`、`d.freq`、`d.<field>` / `d["field"]`、`d.m1`、`d.agg(...)`、`d.save_field(...)`。
- **外部信号对齐**：`d.align(ts, values)` 把带“可知时刻”的外部序列对齐到 bar 网格——可知时刻落在 `(bar 开盘, bar 收盘]` 的值归入该 bar，同一 bar 多个值取最后一个，无值为 NaN（可选 `ffill`）。

---

## 7. 算子库 `tsbt.ops`

普通 Python 函数，输入输出都是 ndarray，**沿第 0 轴（时间）计算，各列互相独立**，`(T,)` 与 `(T, N)` 通用。同名算子与 AlphaForge 语义一致，窗口参数 `d` 的单位是 **bar 数**（不是天）。

统一的 NaN 语义（沿用 AlphaForge）：

- 滚动窗口最小有效样本数 `min_valid(d) = min(d, max(2, ⌈d/2⌉))`，不足则输出 NaN；窗口内 NaN 被跳过。
- 标准差、协方差用 `ddof=1`；偏度、峰度用总体中心矩。
- 所有非有限结果（±inf、除零）统一转 NaN。

| 类别 | 算子 |
|---|---|
| 逐元素 | `add subtract multiply divide inverse reverse abs sign log exp sqrt power signed_power tanh sigmoid clamp max min round round_down floor ceiling fraction if_else greater greater_equal less less_equal equal not_equal and_ or_ not_ is_nan is_not_nan is_finite nan_mask nan_out to_nan purify left_tail right_tail` |
| 位移 / 填充 | `ts_delay ts_delta ts_returns log_diff ts_backfill` |
| 滚动统计 | `ts_sum ts_mean ts_std_dev ts_min ts_max ts_median ts_rank ts_zscore ts_scale ts_av_diff ts_ir ts_arg_max ts_arg_min ts_product ts_skewness ts_kurtosis ts_moment ts_quantile ts_entropy ts_count_nans inst_tvr` |
| 加权 / 滤波 | `ts_decay_linear ts_decay_exp_window filter` |
| 双序列 | `ts_corr ts_covariance ts_regression ts_seq_beta` |
| 递推（numba） | `ts_ema ts_sma_cn hump hump_decay jump_decay trade_when keep` |

说明：

- ndarray 本身支持 `+ - * /`，逐元素算子的价值在于 NaN / 除零语义统一和名称对齐。
- **唯一的语义差异**：AlphaForge 的 `hump` 以“当日截面 |x| 均值 × hump”为阈值，会把各标的耦合在一起。tsbt 的 `hump(x, threshold)` 改为**绝对阈值**，各列独立；`hump_decay` 语义不变。
- 未移植：全部截面算子、`group_*`、`fund_*`、`vec_*`、`bucket`、`densify`。

---

## 8. 回测引擎

### 8.1 接口

```python
res = bt.run(
    signal, data,
    price="vwap",          # open | twap | vwap | close
    exec_window=None,      # twap/vwap 的成交窗口（分钟），None = 整根 bar
    fee=2e-4,              # 每单位成交名义的费率（手续费 + 滑点）
    compound=True,         # True 复利 / False 单利
    delay=1,               # 信号到成交的 bar 延迟，必须 >= 1
    mode=None,             # "independent" | "portfolio"；None: N=1 独立，N>1 组合
    alloc=None,            # 组合资金分配，见 8.6；None = 等权
    nan_policy="flat",     # 信号 NaN: "flat" 平仓 | "hold" 沿用上一目标
    start=None, end=None,  # 评估区间
)
```

`signal` 形状须与 `data` 的 `(T,)` / `(T, N)` 一致；形状不符直接报错。也接受 `(T, K)` 信号配单标的数据（K 组参数批量回测，`mode="independent"`）。

### 8.2 仓位生成

```
θ[t] = alloc[t-delay] ⊙ signal[t-delay]        # t < delay 时 θ = 0
```

`θ[t]` 是 bar t 成交后持有的目标仓位。最后 `delay` 行信号没有可成交的 bar，不参与回测。

### 8.3 记号

对每个标的 i、每根 bar t：

```
P[t]  执行价（由 price / exec_window 决定，取自 bar t 内部）
C[t]  bar t 收盘价
R[t]  = P[t+1] / P[t] − 1      相邻两次成交之间的收益（持有 θ[t] 期间）
b[t]  = C[t]  / P[t] − 1       成交后到本 bar 收盘的收益
f     费率
```

净值统一**按 bar 收盘价盯市**，时间戳为 bar 收盘时刻。这样四种执行价的净值在同一时间网格上，可直接互相比较，也可与基准比较。

### 8.4 复利模式（`compound=True`）

`θ` = 持仓名义价值 / 当前权益。

```
G[t]      = 1 + Σ_i θ[t,i]·R[t,i]                    成交 t → 成交 t+1 的权益增长
w⁻[t+1,i] = θ[t,i]·(1 + R[t,i]) / G[t]               下次成交前的漂移权重，w⁻[0] = 0
τ[t]      = Σ_i |θ[t,i] − w⁻[t,i]|                   bar t 的双边换手
E⁺[t]     = Π_{u≤t}(1 − f·τ[u]) · Π_{u<t} G[u]       bar t 成交后的权益
NAV[t]    = E⁺[t] · (1 + Σ_i θ[t,i]·b[t,i])          按 bar t 收盘盯市
```

费前净值 `NAV_gross` 取 `f = 0`。全部是 `cumprod` 与逐元素运算，无时间循环。

### 8.5 单利模式（`compound=False`）

`θ` = 持仓名义价值 / 固定本金 1，盈亏不再投入。

```
pnl[t]    = Σ_i θ[t,i]·R[t,i]
h⁻[t+1,i] = θ[t,i]·(1 + R[t,i])                      漂移后的持仓名义，h⁻[0] = 0
τ[t]      = Σ_i |θ[t,i] − h⁻[t,i]|
NAV[t]    = 1 + Σ_{u<t} pnl[u] − f·Σ_{u≤t} τ[u] + Σ_i θ[t,i]·b[t,i]
```

### 8.6 漂移换手数值例

信号连续两期都是 0.2，其间执行价上涨 5%：

| 模式 | 漂移后仓位 | 换手 | 费用（万 2） |
|---|---|---|---|
| 复利 | `0.2 × 1.05 / (1 + 0.2 × 0.05) = 0.207921` | `0.007921` | `1.58e-6` |
| 单利 | `0.2 × 1.05 = 0.21` | `0.01` | `2.0e-6` |

该数值例直接进入场景测试。

### 8.7 多标的

**`mode="independent"`**：每列是一个独立账户，§8.4 / §8.5 的 `Σ_i` 去掉，按列向量化。返回 N 个单标的结果。

**`mode="portfolio"`**（共享资金池）：`θ[t,i] = alloc ⊙ signal` 是占**组合总权益**的仓位，漂移权重的分母用组合整体增长 `G[t]`，换手与费用按真实调仓量计算，等同一个账户同时交易多个标的。返回：

- 组合结果（净值、指标、图）；
- 各标的的**独立回测结果**（同参数、各自为一个账户、各自对比自己的标的），用于明细表；
- 各标的对组合的盈亏与换手贡献。

**资金分配 `alloc`**（行 `t` 与信号同样视为“截至 bar t 收盘可知”，由引擎统一延迟）：

| 写法 | 含义 |
|---|---|
| `None` / `bt.alloc.equal()` | 静态等权 `1/N`（默认） |
| `bt.alloc.equal_active()` | 在当期有行情的标的间等权 `1/N_active(t)` |
| `bt.alloc.fixed(w)` 或 `(N,)` 数组 | 静态指定权重 |
| `(T, N)` 数组 | 任意时变权重 |
| `bt.alloc.inv_vol(window)` | 反比于各标的策略收益的滚动波动 |
| `bt.alloc.rolling_sharpe(window, floor=0.0)` | 正比于各标的策略滚动 Sharpe，低于 `floor` 截断 |
| 可调用 `f(ctx) -> (T, N)` | 完全自定义 |

`ctx` 提供：`ts / symbols / signal / data`，以及各标的独立回测的 `ret`、`ret_gross`（`(T, N)`，行 t 为 bar t 收盘时已实现的收益）。内置分配器保证因果；自定义函数只应使用 `≤ t` 的行。通用选项：`normalize=True`（每行归一到和为 1，默认）、`rebalance_every=k`（分配权重每 k 根 bar 更新一次，其间保持不变）。

### 8.8 缺失值与边界

| 情况 | 处理 |
|---|---|
| 信号 NaN（如算子预热期） | `nan_policy="flat"` → 目标 0；`"hold"` → 沿用上一目标 |
| 标的上市前 / 下市后 | 目标强制为 0；下市时按最后一根有效 bar 的执行价平仓并计费 |
| 区间中部某根 bar 无价格 | 该 bar 不成交：目标沿用上一期，价格前向填充（期间收益为 0，恢复交易时一次性体现） |
| 评估起点 `start`（默认） | 第一根有非 NaN 目标的 bar 的开盘时刻，此时净值 = 1 |
| 复利模式权益 ≤ 0 | 净值置 0 并终止，结果标记 `bankrupt=True` |
| 费用与目标权重 | 费用按权益的 `f·τ` 比例扣除，成交后权重记为 `θ`（忽略费用对权重的二阶影响，量级为 `f·τ`） |

### 8.9 基准

基准是**持有交易标的本身**，且**不依赖策略自身的表现**。单标的沿用“同口径、信号恒为 1、费率 0”；组合用**买入不动**的篮子。

- 单标的复利：恰为买入持有 `C[t] / C[start]`。
- 单标的单利：标的简单收益的累加，与策略的单利记账口径一致。
- 组合：**买入不动**。每个标的在组合起点（晚上市的在其上市的第一根 bar）按权重买入一次，之后不再调仓，权重随价格漂移；等待上市的份额和权重和不足 1 的部分是现金；数据提前结束的标的按最后一根 bar 的执行价卖出。复利、单利两种口径下数值相同。初始权重：分配是静态的（等权、`fixed`、`(N,)` 数组）时用同一权重；分配随时间变化（滚动 Sharpe、反波动、`(T, N)` 数组、自定义函数）时，权重可能来自策略表现，改用等权 `1/N`。
- 注意它不是“每根 bar 调回固定权重”的篮子：后者自带再平衡收益（BTC/ETH/SOL 1h、2020-01 起实测，期末 33.2x 对 22.6x），且不是买了放着就能拿到的收益。

---

## 9. 指标

基于**费后**净值。风险收益类指标先把 bar 净值按 UTC 日采样（取每个 UTC 零点前最后一个 bar 收盘净值，首尾不足一天的部分各算一个观测），得到日收益 `r_d`；`mdd` 用 bar 级净值。一年按 365 天。

| 指标 | 复利模式 | 单利模式 |
|---|---|---|
| `start` | 评估区间第一根 bar 的开盘时刻 | 同 |
| `end` | 评估区间最后一根 bar 的收盘时刻 | 同 |
| `days` | `(end − start)` 的天数 | 同 |
| `apr` | 年化复合收益 `NAV_end^(365/days) − 1` | `(NAV_end − 1) × 365 / days` |
| `sharpe` | `mean(r_d) / std(r_d, ddof=1) × √365`，无风险利率 0 | 同，`r_d` 为日盈亏 / 本金 |
| `mdd` | `min_t (NAV[t] / max_{u≤t} NAV[u] − 1)`，负数 | `min_t (NAV[t] − max_{u≤t} NAV[u])`，负数，以本金为单位 |
| `calmar` | `apr / |mdd|` | 同 |
| `turnover` | **日均双边换手** `Σ_t τ[t] / days` | 同 |

附带输出（不在必需清单内，零成本）：`total_return`、`apr_gross`、`sharpe_gross`、`fee_drag`（= `turnover × 365 × fee`）、`avg_abs_pos`、`long_ratio`，以及基准的 `apr / sharpe / mdd`。

`res.summary()` 返回 polars DataFrame：单标的 1 行；组合为“组合 1 行 + 各标的各 1 行”。`res.yearly()` 返回按自然年分段的同一组指标。

---

## 10. 图表与落盘

图表全部用 `xy.pyplot`（原生模式）绘制，输出 PNG；绘图代码集中在 `plot.py` 一个文件内。xy 目前是 alpha（0.0.9），版本钉在 `>=0.0.9,<0.1`。

> 已在本机冒烟验证 xy 0.0.9：双面板共享 x 轴 + `height_ratios`、`fill_between`、日期轴、`imshow` + 文字标注、柱状图、`twinx`、`savefig` 导出 PNG 均可用，且导出不需要浏览器。

| 文件 | 内容 |
|---|---|
| `nav.png` | **上**：费后净值、费前净值、标的基准（三条线，起点归一）。**下**：费后净值的回撤面积图 |
| `position.png` | **上**：标的收盘价。**中**：持仓仓位（组合为净敞口与总敞口）。**下**：每日换手柱 |
| `long_short.png` | 多头仓位贡献、空头仓位贡献、合计三条累计曲线（费前，按每期贡献 `θ·R` 累加） |
| `periodic.png` | **上**：月度收益热力图（年 × 月，格内标数值）。**下**：年度收益柱 + 当年 Sharpe / MDD 标注 |
| `rolling_sharpe.png` | 滚动 Sharpe（日频，默认 90 天窗口，可配），附基准的滚动 Sharpe 作参照 |

```python
res.plot()                 # 5 张图的容器；作为 notebook 单元格的最后一行会逐张显示
res.plot("nav")            # 单张
res.save("runs/mom_4h")    # 落盘
```

`res.save(dir)` 写入：

```
metrics.json      指标（含基准）
config.json       全部回测参数、标的、频率、数据区间、tsbt 版本
series.parquet    ts / nav / nav_gross / benchmark / ret / turnover / fee / 各标的仓位
yearly.parquet    年度分段指标
*.png             上表 5 张图（组合模式另有 symbols/<SYMBOL>/nav.png）
```

---

## 11. 端到端示例

```python
import numpy as np
import tsbt as bt
from tsbt import ops as op

# 单标的
d = bt.load("BTCUSDT", freq="4h", fields=["close", "volume", "oi"])
mom = op.ts_zscore(op.ts_returns(d.close, 6), 180)
sig = op.tanh(mom) * 0.5 * op.less(op.ts_zscore(d.oi, 180), 2.0)

res = bt.run(sig, d, price="vwap", fee=2e-4)
print(res.summary())
res.save("runs/btc_mom_4h")

# 多标的共享资金池，按滚动 Sharpe 分配
syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
d = bt.load(syms, freq="1h")
sig = op.tanh(op.ts_zscore(op.ts_returns(d.close, 24), 720)) * 0.5
res = bt.run(sig, d, price="twap", exec_window=15,
             alloc=bt.alloc.rolling_sharpe(24 * 90, floor=0.0, rebalance_every=24))

# 外部算好的信号
ext = d.align(my_ts, my_values, ffill=True)
res = bt.run(ext, d, price="open", compound=False)
```

---

## 12. 工程结构

```
tsbt/
├── pyproject.toml            uv 管理，src 布局，Python 3.12
├── SPEC.md
├── src/tsbt/
│   ├── __init__.py           load / load_m1 / run / feature_set / alloc / ops
│   ├── config.py             路径与默认值（环境变量）
│   ├── data/
│   │   ├── catalog.py        数据集注册 + 字段目录
│   │   ├── sources.py        源文件发现、monthly/daily 取舍
│   │   ├── store.py          zarr 存取、清单、文件锁
│   │   ├── m1.py             L1 构建：klines / metrics
│   │   ├── features.py       用户特征集（duckdb SQL / Python）
│   │   ├── bars.py           L2：reshape 归约、执行价、d.agg
│   │   └── api.py            BarData / load / load_m1
│   ├── ops/                  elementwise / rolling（含位移与填充）/ recurrence
│   ├── engine/
│   │   ├── core.py           复利 / 单利 × 独立 / 资金池 的闭式核
│   │   ├── run.py            bt.run：信号 → 仓位 → 盈亏
│   │   ├── alloc.py          资金分配器
│   │   └── result.py         Result / BatchResult / PortfolioResult
│   ├── metrics.py
│   ├── plot.py               唯一依赖 xy 的文件
│   └── cli.py                tsbt cache build / status / clear
└── tests/scenarios/
```

依赖：`numpy`、`polars`、`duckdb`、`zarr>=3`、`bottleneck`、`numba`、`xy>=0.0.9,<0.1`、`filelock`；开发依赖 `pytest`、`ruff`。

---

## 13. 场景测试

只写场景测试，不写琐碎单测。每个测试对应一条真实使用路径或一种真实失败模式；单个测试不超过 10 秒。数据用临时目录里按真实目录结构生成的小型合成 parquet，另有少量读取真实数据的测试（数据根不存在时跳过）。

| # | 场景 | 断言 |
|---|---|---|
| 1 | **成交时点** | 构造只在 bar T 发生跳变的价格；T-1 的信号成交价恰为 T 的 open/twap/vwap/close，绝不是 T-1 的任何价格；`delay=0` 报错 |
| 2 | **未来数据无关性** | 改动 bar ≥ T 的全部数据，bar < T 的仓位、换手、净值逐位不变（数据 → 算子 → 引擎全链路） |
| 3 | **四种执行价手算** | 已知分钟数据的 3 根 bar，四种执行价及 `exec_window=N` 的净值与手算一致 |
| 4 | **漂移换手** | §8.6 数值例：恒定信号 0.2、价格 +5%，复利换手 0.007921、单利 0.01；费前费后净值之差恰为累计费用 |
| 5 | **独立参考模拟器对拍** | 另写一个按“合约张数 + 现金”逐 bar 记账的朴素循环（与权重代数是两套推导），在随机数据上与向量化引擎对拍到 1e-10；覆盖 复利/单利 × 单标的/资金池 × 四种执行价 |
| 6 | **恒等式** | 信号 ≡ 1、费率 0 的复利净值 = 基准 = 买入持有；信号 ≡ 0 净值恒为 1 |
| 7 | **频率合成** | 真实 BTCUSDT 一个月：reshape 合成的 1h/2h/4h/6h/12h/1D OHLCV 与 polars `group_by_dynamic` 独立计算一致；UTC 对齐；首尾不完整 bar 被丢弃 |
| 8 | **缓存** | 第二次加载不触碰源文件且结果逐位相同；新增一天的 daily 文件后只增量追加；monthly/daily 重叠不产生重复 |
| 9 | **metrics 因果对齐** | 时间戳恰在 bar 边界的记录不出现在以该时刻收盘的 bar 中；乱序、重复、非整点记录处理正确 |
| 10 | **用户特征** | 合成逐笔文件上，SQL 表达式、完整查询、Python 可调用三种写法的 1m 值正确、可按声明规则聚合到 bar；改 SQL 只重算该特征集；中断后续跑 |
| 11 | **bar 内聚合** | `d.agg` 的 std / corr / 自定义函数与逐 bar 朴素计算一致；`window=N` 只用前 N 分钟 |
| 12 | **指标** | 已知 CAGR / 回撤 / 波动的构造净值，指标与解析值一致；同一条“日内不变”的策略用 1h 与 1D 回测得到相同的日频 Sharpe |
| 13 | **组合分配** | 静态权重与参考模拟器一致；滚动 Sharpe 分配只依赖过去（改未来收益不改变过去权重）；上市时间不同的标的处理正确 |
| 14 | **算子** | 全部算子的截断不变性（只用过去数据）；带 NaN 的随机输入上与 AlphaForge 对应核一致（AlphaForge 不可导入时跳过） |
| 15 | **报告** | `res.save()` 产出含全部必需键的 `metrics.json` 与 5 张非空 PNG |

---

## 14. 性能目标与实测（2026-11-18，24 核 / 125 GB）

| 操作 | 目标 | 实测 |
|---|---|---|
| 单标的 klines 首次建 L1（约 6.7 年、350 万行） | < 10 s | BTCUSDT 2.6 s；50 个标的共 67 s（平均 1.35 s） |
| 已缓存时加载单标的 4h bar（全部 klines 字段） | < 50 ms | 21 ms（只取 close 14 ms） |
| 已缓存时加载 50 个标的 1h close + vwap | — | 0.92 s |
| 引擎核：单标的 4h 全历史（T ≈ 1.5 万） | < 5 ms | 0.37 ms（`bt.run` 全流程 2.2 ms） |
| 引擎核：50 个标的 1h 全历史（T ≈ 5.9 万） | < 200 ms | 资金池 32 ms，独立账户 90 ms；`bt.run` 含 50 个分标的结果 0.92 s |
| aggTrades 特征集 | 以文件扫描为界 | BTCUSDT 单月（6,400 万行）两个特征集（含窗口函数）2.6 s，峰值内存 4.8 GB；再次加载 7 ms |

## 15. 我先定下的默认口径（请确认或否决）

1. **TWAP 的分钟价格**用该分钟的 VWAP（`quote_volume / volume`，零成交分钟用 close），而不是 close 或 OHLC 均值。
2. **净值按 bar 收盘盯市**（成交前、成交后两段收益分别记账），而不是只在成交时点记净值。
3. **APR**：复利模式用年化复合收益（CAGR），单利模式用算术年化。
4. **turnover = 日均双边换手**（买卖都计入；从 0 开到 0.2 记 0.2）。
5. **mdd 以负数表示**，`calmar = apr / |mdd|`。
6. **信号 NaN 默认平仓**（`nan_policy="flat"`），预热期不持仓。
7. **默认资金分配是静态 `1/N`**；标的上市时间不一时，早期会有闲置资金。想避免可用 `equal_active()`。
8. **基准用同口径的“信号恒为 1”**：复利下即买入持有，单利下为标的简单收益累加。
9. **缓存用 zarr v3，位置默认项目根目录的 `.cache/`**（`/home` 剩余 1.5 TB）。
10. **算子依赖 bottleneck + numba**（递推算子无法纯 numpy 向量化）；回测引擎本身只用 numpy。
11. **`hump` 改为绝对阈值**，与 AlphaForge 的截面缩放语义不同。
12. **metrics 时间戳按“该时刻之后才可知”处理**，bar 边界上的记录归下一根 bar（偏保守 1 分钟）。
13. 把采访预览里的 `bt.bar_agg(x, '4h', how)` 调整为 `d.agg(x, how)`，原因见 §6.6。

---

## 16. 里程碑

| 阶段 | 内容 | 对应测试 |
|---|---|---|
| M1 | 工程骨架（uv）、数据集注册、L1（klines / metrics）、L2、`load`、缓存与 CLI | 7、8、9 |
| M2 | 引擎（复利 / 单利，独立 / 资金池）、参考模拟器、指标 | 1 ~ 6、12 |
| M3 | 算子库 | 14 |
| M4 | `d.agg`、用户特征集（duckdb） | 10、11 |
| M5 | 资金分配器、组合结果 | 13 |
| M6 | xy 图表、`res.save` | 15 |

---

## 17. 实现备注（相对上文的细化）

1. **结果起点**：每个账户从“第一根有可用（已延迟）信号的 bar”开始，这根 bar 开盘时净值 = 1。因此 `res` 的行与 `BarData` 的行可能错开，按 `res.open_ms` / `res.ts` 对齐。
2. **`d.agg`**：分位数写作 `d.agg(x, "quantile", q=0.9)`；`exec_price()` 始终返回 `(T, N)`。
3. **Python 特征集**须给 `outputs=[...]`（或用 `bar_agg` 的键）；`query=` 特征的输出列由 DuckDB 对空表 `DESCRIBE` 推断，不读数据。DuckDB 结果经 `fetchnumpy()` 转 polars，不依赖 pyarrow。分批落盘间隔为 `tsbt.data.m1.FLUSH_SECONDS`（默认 30 s）。
4. **动态分配器的预热**：某行所有已上市标的都还没有权重（历史不足）时，该行用已上市标的等权；否则历史不足的标的权重为 0（新上市标的在积累够历史前不分配资金）。另提供 `bt.alloc.custom(f)`。
5. **组合中区间内缺 bar**：该标的目标沿用上一期；共享资金池里其他标的的漂移仍会让它产生很小的再平衡换手（场景测试中 < 1%），这是“目标沿用上一期”的直接结果。
6. **xy 0.0.9 的限制**：`imshow` 不支持自定义 colormap，热力图颜色在代码中按蓝-灰-红发散色阶计算后以 RGB 图传入；`axhline` 渲染出错，零线用普通线段绘制；`spines.set_color` 不可用。
7. **图表配色**：费后净值蓝、费前橙、基准中性灰、回撤与负值红；不使用双 y 轴（价格、仓位、换手分面板）。高频仓位在 `position.png` 中以浅色逐 bar 填充 + 日均线表示。
8. **非有限值**：信号和分配权重里的 `±inf` 与 NaN 一样视为“无信号 / 不分配”，不会被当成巨大仓位。滚动算子把 `±inf` 输入当作缺失样本。
9. **窗口比样本长**：滚动算子不报错，按截断的头部窗口处理（有效样本达到门槛即出值）。
10. **零方差**：`ts_std_dev` 在低相对波动的窗口上改用精确的两遍算法（bottleneck 的滑动更新在常数窗口上会留下约 √eps × 量级的残差），纯舍入量级的结果记为 0；`ts_corr` / `ts_regression` 的方差低于平方和的 1e-11 时视为零方差。因此常数窗口上 `ts_zscore`、`ts_ir`、`ts_corr` 输出 NaN 而不是噪声。
11. **特征集参数校验**：`bar_agg` / `fill` 的键必须是输出列名，`bar_agg` 的值必须是内置单输入归约名；拼错直接报错（否则会静默退回默认的 `last` / NaN）。
12. **构建中断**：出错或 Ctrl-C 时，已算完的源文件先落盘、排队中的文件不再启动；下次只补未完成的文件。
13. **`names=`**：给信号列命名；组合模式下用于各标的的独立账户。结果的 `config["alloc"]` 记录分配器及其参数。
14. **`bt.alloc.custom(f)`** 与直接传 `f` 含义相同（NaN 行不分配），只是多了 `normalize` / `rebalance_every` 选项。
15. **`save_field(overwrite=True)` 整体替换**已保存的字段：此前保存在当前对象时间范围之外的值会被丢弃，新旧版本不会混在一起。
16. **特征构建的线程上限**：同时处理的文件数 × 每个文件的 DuckDB 线程数（特征集的 `threads`，默认 4）不超过机器核数；`workers` 超出时自动下调。
17. **图表对象**：`res.plot()` 返回 `Figures`（字典子类），在 notebook 里逐张显示；每张图建好即从 pyplot 的全局注册表移除，重复调用不会累积。
18. **`ts_covariance` 的常数窗口**：任一序列在窗口内不动时结果为精确的 0（真值即 0；否则滑动求和会留下符号随机的极小残差）。
19. **源文件被撤回**：某段源文件被删除且没有替代时，除了把那段数据置空，还会从已存数据重新推导该标的的数据起止时间，那段时间不再算作“已上市”。
