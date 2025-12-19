# 选股分析

## SelectorAnalyzerBase

 通用选股策略分析基类

\- 支持：按时间段自动载入 pool_storage 下的 YYYYMMDD.csv

\- 支持：自动判别交易日（优先 pandas_market_calendars；否则 fallback 工作日）

调用：

```python
analyzer = SelectorAnalyzerBase(
    start_date=start_date,				# 起始日
    end_date=end_date,					# 结束日
    analysis_output_dir=analysis_dir,	# 分析结果路径
    code_col="code",					# 股票代码列名
    date_col="select_time",				# 日期列名
    exchange_calendar="XSHG"			# 交易所日历
)

df_selection, df_stats = analyzer.analyze_range(
    pool_dir=pool_dir,
    only_trade_days=True,   # ✅ 自动按交易日过滤/统计
    missing="skip",         # 缺文件先跳过（你也可以改成 "raise" 强制报错）
    plot=True
)
```





## NPatternAnalyzer

分析指定时间段 1板n调 (n=1,2,3)选出来的股票，并构造所需数据集

调用：

```python
# ========= 初始化分析器 =========
analyzer = NPatternAnalyzer(
    start_date=start_date,				# 起始日
    end_date=end_date,					# 结束日
    pool_dir=POOL_DIR,					# 股票池路径
    market_data_dir=MARKET_DIR			# 市场文件路径
)
#构造分析矩阵
analyzer.analyze_data_build( window_days= 10, past_days = 20) 
# window_days : 未来窗口长度
# past_days: 过去窗口长度
```



### 数据集字段说s明

`code`

- **含义**：股票代码（标准化，如 `000001.SZ`）
- **来源**：选股池 / parquet 文件名

`select_time`

- **含义**：选股日期（T 日）
- **定义**：T 日交易日日期（`YYYY-MM-DD`）

`Pattern`

- **含义**：选股形态标签
- **示例**：`1板1调 / 1板2调 / 1板3调`
- **语义**：事件/形态驱动信号来源

`n_adjust`

- **含义**：调整次数（从 Pattern 中解析）
- **示例**：`1板2调 → 2`





---





`board_limit_to_select_close_adj_pct`

- **含义**：板块涨跌停到选股日收盘的相对调整幅度

- 计算：
  $$
  \frac{Limit\_up_{board}-Close_{T}}{Limit\_up_{board}}
  $$
  

`avg_limit_to_close_adj_per`

- **含义**：板块层面“涨停 → 收盘”的平均调整幅度

- 计算：
  $$
  \frac{Limit\_up_{board}-Close_T}{Limit\_up_{board}(T_{board}-T)}
  $$
  

`board_close_to_select_close_adj_pct`

- **含义**：板块收盘到个股选股日收盘的相对偏离

- 计算：
  $$
  \frac{Close_{board}-Close_{T}}{Close_{board}}
  $$
  

`avg_close_to_close_adj_per`

- **含义**：板块收盘→收盘的平均涨跌幅

- 计算：
  $$
  \frac{Close_{board}-Close_{T}}{Close_{board}(T_{board}-T)}
  $$
  

`upper_shadow_T`

- 含义：上影线比例

- 计算：
  $$
  \frac{High_T-\max(Open_T,Close_T)}{Close_T}
  $$
  

`lower_shadow_T`

- 含义：下影线比例

- 计算：
  $$
  \frac{\min(Open_T,Close_T)-Low_T}{Close_T}
  $$
  

`body_T`

- 含义：实体大小

- 计算：
  $$
  \frac{|Close_T-Open_T|}{Close_T}
  $$
  

`range_T`

- 含义：日内振幅

- 计算：
  $$
  \frac{High_T-Low_T}{Close_T}
  $$
  

`close_pos_T`

- 含义：收盘位置

- 计算：
  $$
  \frac{Close_T-Low_T}{High_T-Low_T}
  $$
  

`ret_1d_T`

- 含义：T 日涨跌幅

- 计算：
  $$
  \frac{Close_T-Close_{T-1}}{Close_{T-1}}
  $$
  

`gap_T`

- 含义：隔夜跳空

- 计算：
  $$
  \frac{Open_T-Close_{T-1}}{Close_{T-1}}
  $$
  



---



`man_bias_T(n=5,10,20)`

- 定义：**均线乖离率**，表示当前价格与某个周期（例如 5 日、10 日、20 日）的均线之间的偏离程度。

- 计算：
  $$
  \frac{Close_T-MAn_T}{MAn_T}\\
  MAn_T=\frac{1}{n}\sum_{i=1}^nClose_{T-i}\\
  n= 5,10,20
  $$
  

`ret_nd_T(n=3,5,10)`

- 定义：是 **n 日收益率**，表示当前价格与过去 `n` 天的收盘价相比的涨幅。

- 计算：
  $$
  \frac{Close_T-Close_{T-n}}{Close_{T-n}}
  $$
  

`vol_ratio_5_T`

- 定义：是 **成交量比**，表示当前交易日的成交量与过去 5 天平均成交量的比值。

- 计算：
  $$
  \frac{VolumeT}{Mean(Volume_{T-1\ldots T-5})}
  $$
  

`amt_ratio_5_T`

- 定义：当日成交额与过去 5 日平均成交额的比值

- 计算：
  $$
  \frac{Amount_T}{Mean(Amount_{T-1\ldots T-5})}
  $$
  

`amt_ma5_T`

- 定义：计算的是过去 5 日的成交额均值

- 计算：
  $$
  Mean(Amount_{T-1\ldots T-5})
  $$
  

`volatility_5_T`，`volatility_10_T`

- 定义：是 **过去 n天的波动率**，衡量过去 n 天的价格波动幅度。

- 计算：
  $$
  Std\left(\frac{Close_t-Close_{t-1}}{Close_{t-1}}\right)\quad(t\in\text{最近}n\text{日})
  $$

`atr_5n_T` ，`atr_10n_T`

- 定义：是 **过去 5 天的平均真实波动范围**

- 计算：
  $$
  TR_t=\max\left(High_t-Low_t,|High_t-Close_{t-1}|,|Low_t-Close_{t-1}|\right)\\
  ATR_n=Mean(TR_{\text{最近}n\text{日}}),\quad ATRn_T=\frac{ATR_n}{Close_T}
  $$



`is_limit_up_T`

- 定义：T日是否涨停

`is_limit_down_T`

- 定义：T日是否跌停

`close_to_limit_up_pct_T`

- 定义：T日收盘接近涨停的程度

- 计算：
  $$
  \frac{Limit\_up-Close_T}{Close_T}
  $$
  

`open_t1`, `open_t2`

- **定义**：T+1 / T+2 开盘价

`close_t1`, `close_t2`

- **定义**：T+1 / T+2 收盘价

`future_close_max`, `future_close_min`

- **定义**：T+1 → T+window_days 内 high / low 极值

`future_open_max`, `future_open_min`

- **定义**：T+1 → T+window_days 内 high / low 极值

`future_high_max`, `future_low_min`

- 定义：**从 T+1 开始计数** argmax/argmin+1

`mfe_t1o_pct`

- 定义：最大有利波动

- 计算：
  $$
  \frac{Max(High_{T+1...})-Open_{T+1}}{Open_{T+1}}
  $$

`mae_t1o_pct`

- 定义：最大不利波动

- 计算：
  $$
  \frac{Min(Low_{T+1\ldots})-Open_{T+1}}{Open_{T+1}}
  $$

`days_to_max_close`, `days_to_min_close`

- 定义：从 T+1 开始到未来窗口内最大/最小收盘价的天数

`max_ret_pct_close`, `min_ret_pct_close`

- 定义：从 T+1 开始到未来窗口内最大/最小收盘价的调整比例

- 计算：
  $$
  \frac{Close_{extreme}-Close_{T+1}}{Close_{T+1}}
  $$

`max_gain_ratio_close`, `min_gain_ratio_close`

- 定义：从 T+1 开始到未来窗口内最大/最小收盘价的平均调整比例

- 计算：
  $$
  \frac{ExtremeClose-Close_{T+1}}{Close_{T+1}(T_{extreme}-(T+1))}
  $$

> ### Open 维度（同理）
>
> - `days_to_max_open`
> - `days_to_min_open`
> - `max_ret_pct_open`
> - `min_ret_pct_open`
> - `max_gain_ratio_open`
> - `min_gain_ratio_open`



`ret_t1o_t2c_pct`

- 定义：T+2的收盘与T+1的开盘的调整幅度

- 计算：
  $$
  \frac{Close_{T+2}-Open_{T+1}}{Open_{T+1}}
  $$

`win_t1o_t2c`

- 定义：T+2收盘价格是否符合收益要求的标签

- 计算：
  $$
  1[ret\_t1o\_t2c\_pct>k]
  $$
  

