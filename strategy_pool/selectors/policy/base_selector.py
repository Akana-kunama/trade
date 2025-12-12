# -*- coding: utf-8 -*-
"""
Module: n_pattern_policy.py
Description: N字反包选股策略 (Pro版) - 适配 QuantProject 2.0
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# 引入项目组件
from config import MARKET_DATA_DIR, BASIC_INFO_DIR
from common.data_structs import BarFields
from strategy_pool.selectors.policy.base_selector import SelectorBase

class NPatternSelector(SelectorBase):
    def __init__(self):
        # 策略名，对应 pool_storage/n_pattern_rebound 文件夹
        super().__init__(strategy_name="n_pattern_rebound")
        
        # 基础信息表路径
        self.stock_info_path = BASIC_INFO_DIR / "stock.csv"

    def load_stock_metadata(self):
        """
        加载 stock.csv (复用你的逻辑)
        """
        if not self.stock_info_path.exists():
            print(f"❌ [Error] 找不到 {self.stock_info_path}")
            return set(), pd.DataFrame()

        try:
            # 尝试读取
            try:
                stock_info = pd.read_csv(self.stock_info_path, encoding='utf-8')
            except UnicodeDecodeError:
                stock_info = pd.read_csv(self.stock_info_path, encoding='gbk')

            # 1. 提取6位代码
            # 假设 CSV 里代码列名叫 'order_book_id' 或 'code'，这里做个兼容处理
            code_col = 'order_book_id' if 'order_book_id' in stock_info.columns else 'code'
            if code_col not in stock_info.columns:
                print("⚠️ stock.csv 缺少代码列")
                return set(), pd.DataFrame()
                
            stock_info['code_key'] = stock_info[code_col].astype(str).str[:6]

            # 2. 筛选 Normal (如果有状态列)
            if 'special_type' in stock_info.columns:
                normal_df = stock_info[stock_info['special_type'] == 'Normal']
            else:
                normal_df = stock_info

            # 3. 提取需要的字段
            cols_to_keep = ['code_key', 'symbol', 'sector_code_name', 'industry_name']
            cols_to_keep = [c for c in cols_to_keep if c in stock_info.columns]
            
            info_df = stock_info[cols_to_keep].copy()
            valid_codes = set(normal_df['code_key'].values)
            
            return valid_codes, info_df

        except Exception as e:
            print(f"❌ 读取 stock.csv 失败: {e}")
            return set(), pd.DataFrame()

    def run(self, date=None, save=True):
        """
        N字反包选股策略：
        - date=None 时：默认用每只股票的最新一根K线作为选股日（和原来一致）
        - date 为 'YYYY-MM-DD' 或 datetime 时：以该日（或之前最近的一个交易日）作为选股日
        """
        print(f">>> [Strategy] 启动 {self.strategy_name} ...")

        # 0. 规范化 date 参数
        target_date = None
        if date is not None:
            # 接受字符串 / datetime / Timestamp
            target_date = pd.to_datetime(date).normalize()
            print(f"📅 指定选股日: {target_date.date()}")

        # 1. 加载白名单
        valid_whitelist, stock_info_df = self.load_stock_metadata()

        # 2. 扫描 Parquet 文件
        stock_dir = MARKET_DATA_DIR / "stock_daily"
        files = list(stock_dir.glob("*.parquet"))
        print(f"📋 扫描行情文件数: {len(files)}")

        selected_pool = []

        # 3. 遍历计算
        for file_path in files:
            # 解析代码: 000001.SZ.parquet -> 000001
            file_stem = file_path.stem
            code_key = file_stem[:6]  # 取前6位纯数字

            # [过滤 1] 板块过滤 (主板 + 创业板)
            if not re.match(r'^(60|00|30)', code_key):
                continue

            # [过滤 2] ST / 停牌过滤 (白名单)
            if valid_whitelist and code_key not in valid_whitelist:
                continue

            try:
                # 读取 Parquet
                df = pd.read_parquet(file_path)
                if len(df) < 5:
                    continue  # 数据太少

                # 确保按时间排序，假设 index 为日期/时间
                df.sort_index(inplace=True)

                # ====== ⭐ 核心：根据 date 选取 curr / prev1 / prev2 / prev3 ======
                if target_date is None:
                    # 没指定日期：用最新一根
                    curr_idx = len(df) - 1
                else:
                    # 指定日期：在 index 中寻找 <= target_date 的最后一条
                    idx = df.index.searchsorted(target_date, side='right') - 1
                    if idx < 0:
                        # 该股在 target_date 之前没有任何K线数据
                        continue
                    curr_idx = idx

                # 需要至少 4 根 K 线来构造 prev3 ~ curr
                if curr_idx < 3:
                    continue

                curr = df.iloc[curr_idx]
                prev1 = df.iloc[curr_idx - 1]
                prev2 = df.iloc[curr_idx - 2]
                prev3 = df.iloc[curr_idx - 3]

                # ====== 涨停判断函数 ======
                def is_limit(row):
                    # 容错处理：考虑到浮点数精度，用差值小于 0.01
                    return abs(row[BarFields.CLOSE] - row[BarFields.LIMIT_UP]) < 0.01

                curr_limit = is_limit(curr)
                prev1_limit = is_limit(prev1)
                prev2_limit = is_limit(prev2)
                prev3_limit = is_limit(prev3)

                reason = None

                # 获取近10天（最多）数据：截止到 curr 当天
                start_10 = max(0, curr_idx - 9)  # 最多往前取9根 + 当天 = 10 根
                last_10 = df.iloc[start_10:curr_idx + 1]
                limit_counts = last_10.apply(is_limit, axis=1)  # Boolean Series

                # 为了兼容原来的切片逻辑，这里沿用相同的相对位置写法：
                # - 注意：当 last_10 长度 < 10 时，这些切片会自动变短，sum() 仍然安全。
                n = len(last_10)

                # === 策略逻辑复刻 ===

                # 模式 A: 1板1调 (昨天板，今天断板且不破板开)
                if prev1_limit and not curr_limit:
                    # 原逻辑：limit_counts.iloc[-8:-2].sum() < 2
                    # 这里保持不变，基于相对尾部切片
                    sub = limit_counts.iloc[max(0, n - 8):max(0, n - 2)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev1[BarFields.OPEN]:
                            reason = "1板1调"

                # 模式 B: 1板2调 (前天板，昨今断，不破板开)
                elif prev2_limit and not prev1_limit and not curr_limit:
                    # 原逻辑：limit_counts.iloc[-9:-3].sum() < 2
                    sub = limit_counts.iloc[max(0, n - 9):max(0, n - 3)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev2[BarFields.OPEN]:
                            reason = "1板2调"

                # 模式 C: 1板3调
                elif prev3_limit and not prev2_limit and not prev1_limit and not curr_limit:
                    # 原逻辑：limit_counts.iloc[-10:-4].sum() < 2
                    sub = limit_counts.iloc[max(0, n - 10):max(0, n - 4)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev3[BarFields.OPEN]:
                            reason = "1板3调"

                if reason:
                    # 计算量比 (和过去5日均量相比)，同样基于 curr_idx
                    vol_start = max(0, curr_idx - 5)
                    vol_end = curr_idx  # 不含 curr
                    vol_hist = df[BarFields.VOLUME].iloc[vol_start:vol_end]
                    vol_ma5 = vol_hist.mean() if len(vol_hist) > 0 else 0
                    vol_ratio = round(curr[BarFields.VOLUME] / vol_ma5, 2) if vol_ma5 > 0 else 0

                    selected_pool.append({
                        'code_key': code_key,
                        'code': curr.get(BarFields.CODE, code_key),
                        'Close': curr[BarFields.CLOSE],
                        'Vol_Ratio': vol_ratio,
                        'Pattern': reason,
                        'select_time': curr.name.strftime('%Y-%m-%d')  # 使用 index 作为日期
                    })

            except Exception as e:
                # print(f"Error: {code_key} - {e}")
                continue

        # 4. 合并与保存
        if selected_pool:
            res_df = pd.DataFrame(selected_pool)

            # 合并行业信息
            if not stock_info_df.empty:
                final_df = pd.merge(res_df, stock_info_df, on='code_key', how='left')
            else:
                final_df = res_df

            # 排序
            if 'industry_name' in final_df.columns:
                final_df.sort_values(by=['industry_name', 'Pattern'], inplace=True)

            # 5. 调用父类方法保存
            # save_result 内部可以自己决定用哪天命名文件，如果你希望用 target_date 命名，也可以在这里加参数
            if save:
                self.save_result(final_df,date_str=date)
            else :
                return final_df
        else:
            print(f"[{self.strategy_name}] ⚠️ 选股日无符合条件股票")


# 调试用
if __name__ == "__main__":
    s = NPatternSelector()
    s.run()