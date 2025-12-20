# -*- coding: utf-8 -*-
from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant, xtdata
import random
import time
import math

from config.account_config import QMTConfig
from toolbox.messengers.wechat_bot import WechatBot

class RealTrader:
    def __init__(self):
        self.account_id = QMTConfig.ACCOUNT_ID
        self.path = QMTConfig.MINI_QMT_PATH
        self.session_id = random.randint(10000, 99999)
        
        self.trader = XtQuantTrader(self.path, self.session_id)
        self.acc = StockAccount(self.account_id, "STOCK")
        
        self.bot = WechatBot()
        self._connect()

    def _connect(self):
        """连接 QMT"""
        self.trader.register_callback(XtQuantTraderCallback())
        self.trader.start()
        ret = self.trader.connect()
        if ret == 0:
            print(f"✅ QMT 实盘连接成功 [账号: {self.account_id}]")
            self.trader.subscribe(self.acc)
        else:
            print("❌ QMT 连接失败，请检查路径或是否登录")

    def _get_latest_quote(self, stock_code):
        """获取实时五档行情"""
        xtdata.subscribe_quote(stock_code, period='tick')
        time.sleep(0.3) # 必须等待订阅生效
        tick = xtdata.get_full_tick([stock_code]).get(stock_code)
        if not tick:
            return None
        return tick

    def _get_position(self, stock_code):
        """查询单只股票持仓"""
        positions = self.trader.query_stock_positions(self.acc)
        for p in positions:
            if p.stock_code == stock_code:
                return p
        return None

    # ============================================================
    # 🟢 场景 1: 按金额买入 (最常用，适合 AI 选股)
    # ============================================================
    def buy_by_amount(self, stock_code, target_amount, price_type='limit'):
        """
        :param target_amount: 计划买入金额 (元)
        :param price_type: 'limit' (卖一价), 'market' (最新价)
        """
        tick = self._get_latest_quote(stock_code)
        if not tick:
            print(f"❌ 行情获取失败: {stock_code}")
            return

        # 确定价格策略：买入时，用卖一价 (askPrice[0]) 更容易成交
        if len(tick['askPrice']) > 0 and tick['askPrice'][0] > 0:
            exec_price = tick['askPrice'][0]
        else:
            exec_price = tick['lastPrice']

        if exec_price <= 0: return

        # 计算数量 (向下取整到100)
        can_buy_shares = int(target_amount / exec_price)
        volume = (can_buy_shares // 100) * 100

        if volume < 100:
            msg = f"⚠️ 金额不足买一手: {stock_code} (需 {exec_price*100:.0f}元)"
            print(msg)
            # self.bot.send_text(msg) # 可选是否推送警告
            return

        # 执行
        self._execute_order(stock_code, xtconstant.STOCK_BUY, volume, exec_price, "BuyAmount")

    # ============================================================
    # 🔵 场景 2: 指定价格和手数买入 (适合网格/信号交易)
    # ============================================================
    def buy_fixed(self, stock_code, price, lots):
        """
        :param lots: 手数 (1手=100股)
        """
        volume = lots * 100
        if volume <= 0 or price <= 0:
            print("❌ 参数错误")
            return
            
        self._execute_order(stock_code, xtconstant.STOCK_BUY, volume, price, "BuyFixed")

    # ============================================================
    # 🟠 场景 3: 按比例卖出 (适合止盈止损)
    # ============================================================
    def sell_by_ratio(self, stock_code, ratio):
        """
        :param ratio: 0.5 = 卖出50%
        """
        pos = self._get_position(stock_code)
        if not pos or pos.can_use_volume <= 0:
            print(f"❌ 无可用持仓: {stock_code}")
            return

        available = pos.can_use_volume
        target_vol = int(available * ratio)
        
        # 凑整逻辑
        volume = (target_vol // 100) * 100
        
        # 特殊情况：如果是清仓 (ratio >= 0.99) 或者 零股
        if ratio >= 0.99:
            volume = available # 全卖
        elif volume == 0 and available < 100:
            # 如果剩几十股碎股，想卖必须通过特殊处理(通常系统不允许直接卖碎股，除非全仓)
            print(f"⚠️ 只有 {available} 股，不足一手，暂不卖出")
            return

        if volume <= 0: return

        # 获取价格：卖出时，用买一价 (bidPrice[0]) 更容易成交
        tick = self._get_latest_quote(stock_code)
        if tick and len(tick['bidPrice']) > 0 and tick['bidPrice'][0] > 0:
            exec_price = tick['bidPrice'][0]
        else:
            exec_price = tick['lastPrice'] if tick else 0

        self._execute_order(stock_code, xtconstant.STOCK_SELL, volume, exec_price, "SellRatio")

    # ============================================================
    # 🔴 场景 4: 指定价格和手数卖出 (适合精准减仓)
    # ============================================================
    def sell_fixed(self, stock_code, price, lots):
        volume = lots * 100
        
        # 风控：检查持仓够不够
        pos = self._get_position(stock_code)
        if not pos or pos.can_use_volume < volume:
            print(f"❌ 持仓不足: {stock_code} (可用: {pos.can_use_volume if pos else 0})")
            return

        self._execute_order(stock_code, xtconstant.STOCK_SELL, volume, price, "SellFixed")

    # ============================================================
    # ⚙️ 核心执行层 (底层)
    # ============================================================
    def _execute_order(self, stock_code, action_type, volume, price, remark):
        """真正的下单动作"""
        
        action_str = "买入" if action_type == xtconstant.STOCK_BUY else "卖出"
        amount = price * volume
        
        # 1. 打印日志
        log_msg = f"【实盘{action_str}】{stock_code} | 价格:{price} | 数量:{volume} | 金额:{amount:.2f}"
        print(log_msg)
        
        # 2. 推送微信
        self.bot.send_text(log_msg)
        
        # 3. 发送指令
        # 使用 FIX_PRICE (限价) 是最稳的，用对手价下单能保证成交率
        order_id = self.trader.order_stock(
            self.acc, 
            stock_code, 
            action_type, 
            volume, 
            xtconstant.FIX_PRICE, 
            price, 
            remark, 
            "Python_Auto"
        )
        
        if order_id <= 0:
            err_msg = f"❌ 下单失败，QMT返回ID: {order_id}"
            print(err_msg)
            self.bot.send_text(err_msg)
        else:
            print(f"✅ 委托成功，订单ID: {order_id}")