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
        print(f">>> [Strategy] 启动 {self.strategy_name} ...")

        target_date = None
        if date is not None:
            target_date = pd.to_datetime(date).normalize()
            print(f"📅 指定选股日: {target_date.date()}")

        valid_whitelist, stock_info_df = self.load_stock_metadata()

        stock_dir = MARKET_DATA_DIR / "stock_daily"
        files = list(stock_dir.glob("*.parquet"))
        print(f"📋 扫描行情文件数: {len(files)}")

        selected_pool = []

        # ========= 工具：把 code_key 转成带后缀的标准 code =========
        def code_key_to_full(code_key: str) -> str:
            code_key = str(code_key).zfill(6)
            if code_key.startswith(("60", "68", "90")):
                return f"{code_key}.SH"
            elif code_key.startswith(("00", "30", "20")):
                return f"{code_key}.SZ"
            elif code_key.startswith(("8", "4")):
                return f"{code_key}.BJ"
            else:
                return f"{code_key}.SZ"

        for file_path in files:
            file_stem = file_path.stem
            code_key = file_stem[:6]

            # 主板 + 创业板过滤
            if not re.match(r'^(60|00|30)', code_key):
                continue

            if valid_whitelist and code_key not in valid_whitelist:
                continue

            try:
                df = pd.read_parquet(file_path)
                if len(df) < 5:
                    continue

                df.sort_index(inplace=True)

                # 根据 date 选 curr
                if target_date is None:
                    curr_idx = len(df) - 1
                else:
                    idx = df.index.searchsorted(target_date, side='right') - 1
                    if idx < 0:
                        continue
                    curr_idx = idx

                if curr_idx < 3:
                    continue

                curr = df.iloc[curr_idx]
                prev1 = df.iloc[curr_idx - 1]
                prev2 = df.iloc[curr_idx - 2]
                prev3 = df.iloc[curr_idx - 3]

                def is_limit(row):
                    return abs(row[BarFields.CLOSE] - row[BarFields.LIMIT_UP]) < 0.01

                curr_limit = is_limit(curr)
                prev1_limit = is_limit(prev1)
                prev2_limit = is_limit(prev2)
                prev3_limit = is_limit(prev3)

                reason = None

                start_10 = max(0, curr_idx - 9)
                last_10 = df.iloc[start_10:curr_idx + 1]
                limit_counts = last_10.apply(is_limit, axis=1)
                n = len(last_10)

                # 1板1调
                if prev1_limit and not curr_limit:
                    sub = limit_counts.iloc[max(0, n - 8):max(0, n - 2)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev1[BarFields.OPEN]:
                            reason = "1板1调"

                # 1板2调
                elif prev2_limit and not prev1_limit and not curr_limit:
                    sub = limit_counts.iloc[max(0, n - 9):max(0, n - 3)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev2[BarFields.OPEN]:
                            reason = "1板2调"

                # 1板3调
                elif prev3_limit and not prev2_limit and not prev1_limit and not curr_limit:
                    sub = limit_counts.iloc[max(0, n - 10):max(0, n - 4)]
                    if sub.sum() < 2:
                        if curr[BarFields.CLOSE] >= prev3[BarFields.OPEN]:
                            reason = "1板3调"

                if reason:
                    # ====== 量比 ======
                    vol_start = max(0, curr_idx - 5)
                    vol_end = curr_idx
                    vol_hist = df[BarFields.VOLUME].iloc[vol_start:vol_end]
                    vol_ma5 = vol_hist.mean() if len(vol_hist) > 0 else 0
                    vol_ratio = round(curr[BarFields.VOLUME] / vol_ma5, 2) if vol_ma5 > 0 else 0

                    # ====== 关键修复：为不同 pattern 选对“板日”bar ======
                    if reason == "1板1调":
                        board_bar = prev1
                        n_adjust = 1
                    elif reason == "1板2调":
                        board_bar = prev2
                        n_adjust = 2
                    else:  # 1板3调
                        board_bar = prev3
                        n_adjust = 3

                    # ====== 统一 code 输出：确保 parquet 可对齐 ======
                    code_full = code_key_to_full(code_key)

                    select_close = float(curr[BarFields.CLOSE])
                    board_limit = float(board_bar[BarFields.LIMIT_UP])
                    board_close = float(board_bar[BarFields.CLOSE])

                    # 你要的两类调整幅度
                    # 1) 板日->选股日 总调整幅度（这里用“板日开盘 board_limit”作为基准价）
                    board_limit_to_select_close_adj_pct = (select_close - board_limit) / board_limit * 100 if board_limit != 0 else np.nan
                    board_close_to_select_close_adj_pct = (select_close - board_close) / board_close * 100 if board_limit != 0 else np.nan
                    # 2) 平均每调幅度
                    avg_limit_to_close_adj_per = board_limit_to_select_close_adj_pct / n_adjust if n_adjust else np.nan
                    avg_close_to_close_adj_per = board_close_to_select_close_adj_pct / n_adjust if n_adjust else np.nan


                    selected_pool.append({
                        "code_key": code_key,
                        "code": code_full,  # ✅ 固定输出 000001.SZ/SH

                        # ===== 选股日信息 =====
                        "select_time": curr.name.strftime("%Y-%m-%d"),
                        "Close": select_close,

                        # ===== 板日信息（新增，口径明确）=====
                        "board_date": board_bar.name.strftime("%Y-%m-%d"),
                        "board_limit": board_limit, # 板日涨停价格
                        "board_close": board_close, # 板日收盘价格

                        # ===== 你关心的指标（直接落表）=====
                        "Pattern": reason,
                        "n_adjust": n_adjust,
                        "board_limit_to_select_close_adj_pct": board_limit_to_select_close_adj_pct,
                        "avg_limit_to_close_adj_per": avg_limit_to_close_adj_per,
                        "board_close_to_select_close_adj_pct": board_limit_to_select_close_adj_pct,
                        "avg_close_to_close_adj_per": avg_close_to_close_adj_per,

                        # # ===== 其它原有字段 =====
                        "Vol_Ratio": vol_ratio,
                    })

            except Exception:
                continue

        # 合并与保存
        if selected_pool:
            res_df = pd.DataFrame(selected_pool)

            # 合并行业信息
            if not stock_info_df.empty:
                final_df = pd.merge(res_df, stock_info_df, on="code_key", how="left")
            else:
                final_df = res_df

            # 排序
            if "industry_name" in final_df.columns:
                final_df.sort_values(by=["industry_name", "Pattern"], inplace=True)

            if save:
                self.save_result(final_df, date_str=date)

            return final_df

        return pd.DataFrame()


# 调试用
if __name__ == "__main__":
    s = NPatternSelector()
    s.run()