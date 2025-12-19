# -*- coding: utf-8 -*-
"""
Module: base_selector.py
Description: 选股策略基类 - 处理路径管理和结果存储
"""
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import STRATEGY_WORKSPACE

class SelectorBase:
    def __init__(self, strategy_name):
        """
        初始化策略
        :param strategy_name: 策略唯一英文名称 (作为文件夹名)
        """
        self.strategy_name = strategy_name
        # 自动定位输出目录: strategy_pool/selectors/pool_storage/{strategy_name}
        self.output_dir = STRATEGY_WORKSPACE / strategy_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, date=None):
        """
        子类必须实现此方法
        :param date: 指定运行日期 (datetime 或 str)，默认为当天
        """
        raise NotImplementedError

    def save_result(self, df_result, date_str=None):
        """
        统一保存选股结果（文件名只保留年月日 YYYYMMDD）
        """
        if df_result is None or df_result.empty:
            print(f"[{self.strategy_name}] 今日无选股结果，跳过保存。")
            return

        # ---- 强制格式化 date_str 为 YYYYMMDD ----
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        else:
            # 自动处理 datetime / timestamp / str 三种输入类型
            date_obj = pd.to_datetime(date_str)
            date_str = date_obj.strftime('%Y%m%d')

        # 增加策略名和日期列
        df_result['strategy_name'] = self.strategy_name
        df_result['date'] = date_str

        # 文件名只会是 YYYYMMDD.csv
        file_path = self.output_dir / f"{date_str}.csv"

        df_result.to_csv(file_path, index=False, encoding='utf-8-sig')

        print(f"✅ [{self.strategy_name}] 结果已保存: {file_path}")
        print(f"   入选数量: {len(df_result)}")