# config/account_config.py
# -*- coding: utf-8 -*-

# ================= QMT 实盘/数据配置 =================
class QMTConfig:
    # QMT 客户端安装路径 (MiniQMT 需要用到 userdata_mini)
    MINI_QMT_PATH = r"E:\Program_Files\中金财富QMT个人版交易端\userdata_mini"
    
    # 账号信息
    ACCOUNT_ID = "0000000"
    ACCOUNT_TYPE = "STOCK"  # STOCK, FUTURE, CREDIT
    
    # 是否开启实盘模式 (False 则只做数据下载或模拟)
    ENABLE_TRADING = True

# ================= Tushare 数据配置 =================
class TushareConfig:
    # 你的 Tushare Token
    TOKEN = "YOUR_TUSHARE_TOKEN_HERE" 

# ================= 其他数据源配置 =================
# 比如 Qlib, AkShare 等可以后续加在这里
class WechatConfig:
    # 企业微信机器人的 Key (敏感信息放这里)
    WEBHOOK_KEY = "238495d7-921a-4b33-8835-ce49cbe3835d"