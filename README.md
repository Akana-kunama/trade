# QuantProject 2.0 - AI 驱动的工程化量化交易系统

[![System Status](https://img.shields.io/badge/System-Active_Development-brightgreen)](https://github.com/yourusername/QuantProject)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Broker Interface](https://img.shields.io/badge/Broker-QMT_(Mini)-orange)](http://www.thinktrader.net/)

**QuantProject 2.0** 是一个基于 Python 构建的现代化、模块化量化交易框架。它打破了传统脚本式量化的局限，引入了 **MLOps（机器学习运维）** 理念，实现了从数据清洗、特征工程、模型训练、策略决策到实盘交易的全链路自动化闭环。

---

## 🌟 核心特性 (Key Features)

### 1. 🏗️ 存算分离的因子工厂 (Factor Lab)
* **标准化定义**：通过继承 `BaseFactor` 即可定义因子，支持 Pandas/Numpy 向量化计算。
* **增量更新引擎**：智能识别数据时间戳，仅计算新增数据，极大降低每日维护耗时。
* **特征仓库 (Feature Store)**：计算结果落盘为 Parquet，支持按列极速读取，彻底解决重复计算痛点。

### 2. 🤖 AI 模型流水线 (Model Factory)
* **多模型集成**：内置 LightGBM, XGBoost, CatBoost 三大树模型封装，开箱即用，支持拓展。
* **模型仓库 (Model Zoo)**：自动管理模型版本，保存训练元数据（特征列表、时间跨度、性能指标），支持一键加载最新模型。
* **配置驱动**：数据加载、特征筛选、标签定义完全通过配置解耦。

### 3. 🛡️ 稳健的工程架构
* **多源异构数据支持**：通过适配器模式（Adapter）统一管理 QMT、Tushare 等不同数据源。
* **A股特色适配**：内置精确的A股涨跌停价格计算（处理特殊的四舍五入精度问题）及 ST 股/科创板/非交易日过滤逻辑。
* **标准化数据流**：全链路采用 Parquet 存储与标准 `BarData` 结构，确保回测与实盘的一致性。

### 4. 策略执行与工具
* **选股策略逻辑**：支持选股策略自定义，选股策略对应股票池。
* **择时策略逻辑**：支持择时策略自定义，选股策略与择时逻辑互相组合。
* **生产级交易**：支持按金额/手数买入、按比例/手数卖出；下单前自动检查持仓可用与价格偏离度。
* **实时监控**：集成企业微信/钉钉机器人，实时推送交易指令与选股日报。

---

## 📂 目录结构说明

```text
QuantProject/
├── config/                     # [配置中心]
│   ├── account_config.py       # 账号、Token、实盘/回测开关 (敏感信息)
│   ├── path_config.py          # 自动路径定位 (无需手动修改)
│   └── system_config.yaml      # 系统级参数
├── common/                     # [基础设施]
│   ├── data_structs.py         # 标准数据结构定义 (BarData, Order)
│   └── constants.py            # 全局常量 (BarFields)
├── data_center/                # [数据工厂] ETL层
│   ├── collectors/             # 数据下载适配器 (QMT/Tushare)
│   ├── storage/                # 原始行情 (不可变数据, Parquet)
│   │   ├── market_data/        # 日线/分钟线行情
│   │   └── basic_info/         # 静态基础表 (stock.csv)
│   └── data_proxy.py           # 统一数据读取接口
├── factor_lab/                 # [因子工厂] Feature Engineering
│   ├── definitions/            # 因子逻辑 (技术指标, Label, 形态)
│   ├── engine/                 # 增量计算与落盘引擎
│   ├── service/                # 对外统一取数接口 (FeatureService)
│   └── storage/                # 因子特征库 (stock_factors/*.parquet)
├── model_factory/              # [模型工厂] AI Core
│   ├── data_loader.py          # 数据集构建 (自动处理Index/Filter)
│   ├── registry.py             # 模型保存、加载、版本控制
│   ├── models/                 # 算法封装 (LGBM/XGB/Cat)
│   └── model_zoo/              # 训练产物归档 (按实验名分类)
├── strategy_pool/              # [策略大脑] Decision Making
│   ├── selectors/              # 选股模块
│   │   ├── policy/             # 策略逻辑代码 (如: NPatternSelector)
│   │   └── pool_storage/       # 策略产出结果 (CSV股票池)
│   └── timers/                 # 择时/交易信号模块
├── engines/                    # [执行引擎] Execution
│   ├── backtest/               # 回测引擎
│   └── trading/                # 实盘交易引擎 (RealTrader, 微信Bot)
├── scripts/                    # [任务脚本] 系统的入口 (Crontab 调度)
└── toolbox/                    # [工具箱] 消息推送、监控、报表🚀 快速开始 (Quick Start)
1. 环境准备
推荐使用 Anaconda 管理 Python 环境（建议 Python 3.8+）。

Bash

# 安装依赖
pip install -r requirements.txt
注意：如果你使用 QMT 作为数据源/交易接口，你需要引用 QMT 自带的 Python 库 (xtquant)。请确保你的 Python 环境能加载 xtquant。

2. 配置项目
打开 config/account_config.py。

配置你的 QMT 安装路径（MINI_QMT_PATH）、账号信息以及微信推送 Key。

确保 data_center/storage/basic_info/ 下存在 stock.csv（基础股票信息表）。

3. 标准工作流 (Workflow)
本系统设计为脚本驱动，适合部署在服务器或本地通过 Task Scheduler 运行。

阶段一：每日收盘后 (T+0 盘后)
更新基础数据 下载最新的日线行情数据，自动计算并清洗涨跌停状态。

Bash

python scripts/run_daily_data.py
更新因子库 基于新行情，增量计算 MA5、量比、N字反转信号及 Label，并更新 Metadata。

Bash

python scripts/run_daily_factors.py
运行选股策略 加载 Model Zoo 中最新的模型，预测明日目标，生成 CSV 并推送微信。

Bash

python scripts/run_daily_selection.py
阶段二：模型迭代 (周末/定期)
当积累了足够多的新数据后，重新训练 AI 模型以适应市场变化。

训练模型：该脚本会自动训练设定 Label 下的多种算法模型，并保存到 model_factory/model_zoo/。

Bash

python scripts/train_n_pattern.py
阶段三：实盘交易 (T+1 盘中)
自动交易：读取昨晚生成的选股 CSV，自动执行买入操作；或根据策略执行卖出。

Bash

python scripts/run_live_trading.py
🛠️ 二次开发指南 (Developer Guide)
1. 如何增加一个新的因子？
无需修改引擎代码，只需在 factor_lab/definitions/ 下新建 Python 文件：

Python

from .base import BaseFactor

class MyNewFactor(BaseFactor):
    name = "rsi_14"
    description = "14日RSI指标"
    
    def compute(self, df):
        # 实现你的计算逻辑，返回 Series
        return rsi_series
然后在 definitions/__init__.py 中 import 它。下次运行 run_daily_factors.py 时，系统会自动计算并存储该因子。

2. 如何扩展其他数据源 (如 Tushare)？
在 data_center/collectors/ 下新建 adapter_tushare.py。

实现 ETL 逻辑，确保输出的 DataFrame 列名符合 common.data_structs.BarFields 定义的标准（open, high, low, close, volume, adj_factor 等）。

在 scripts/run_daily_data.py 中引入并调用该 Adapter。

3. 如何增加一个新的选股策略？
在 strategy_pool/selectors/policy/ 下新建 my_strategy.py 并继承 SelectorBase。

在 run() 方法中调用 FeatureService 获取数据，或调用 ModelRegistry 加载模型进行预测。

在 scripts/run_daily_selection.py 中注册运行。

⚠️ 注意事项与免责声明
涨跌停精度：A股涨跌停计算采用特殊的四舍五入规则（非银行家舍入），本框架在 adapter_qmt.py 中已做特殊处理，请勿随意修改相关算法。

数据一致性：run_daily_data.py 默认采用覆盖更新模式（最近N天），适合中低频策略。如需更高性能，可自行改为 Append 模式。

风险提示：

本系统仅供学习与研究使用，实盘交易存在巨大风险。

实盘模块依赖 XtQuant 库及 QMT 客户端，请确保环境配置正确且账号已登录。

AI 模型预测基于历史数据，不代表未来收益保证。

📝 TODO List


[ ] Web 看板：开发 Streamlit 界面，可视化展示因子覆盖率与模型 AUC 曲线

[ ] 基本面因子：接入财务数据（PE/PB/ROE），并在 Engine 层实现 merge_asof 对齐

[ ] 数据源接入：接入tushare，qlib等数据源

[ ] 盘中风控：增加监控股票的实盘qmt程序，实现盘中实时监控

[ ] 策略拓展：目前主要完成的N字反转的启动策略，需要拓展N字反转拓展的反包策略，即选出N字之后的跌停在T+1或T+2的反转

[ ] 策略拓展：N字反转也会挖掘到二波策略，即之前多次涨停，目前是在选股层面过滤，可以考虑加入作为分支