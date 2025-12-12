# -*- coding: utf-8 -*-
"""
Module: n_pattern_analyzer.py
Description: 对 NPatternSelector 的历史选股数据做统计分析 + 可视化
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt   # 👈 新增

from strategy_pool.selectors.policy.n_pattern_policy import NPatternSelector


class SelectorAnalyzer:
    def __init__(self, selector_cls, start_date, end_date, analysis_output_dir=None):
        """
        :param selector_cls: 选股策略类（比如 NPatternSelector）
        :param start_date:   字符串或 datetime, 如 '2014-01-01'
        :param end_date:     字符串或 datetime, 如 '2025-12-31'
        :param analysis_output_dir: 分析结果输出目录，默认在 strategy 的 output_dir 下建一个 analysis 子目录
        """
        self.selector_cls = selector_cls
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # 先实例化一个 selector，用来访问它的 output_dir
        tmp_selector = self.selector_cls()
        base_output_dir = tmp_selector.output_dir

        if analysis_output_dir is None:
            self.analysis_output_dir = base_output_dir / "analysis"
        else:
            self.analysis_output_dir = Path(analysis_output_dir)

        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
    def run_count_per_day(self, overwrite=False, freq="B", plot=True, save_selection=False):
        """
        在 [start_date, end_date] 区间内，按交易日（工作日）循环跑 selector，
        统计每天入选股票数量。

        :param overwrite:      保留兼容性，如果以后你又想用“先读已有 CSV”的模式再说；
                                当前 save_selection=False 时，这个参数其实可以忽略。
        :param freq:           日期频率，默认 'B' = Business day（工作日）
        :param plot:           是否画图
        :param save_selection: 是否在每天 run 时保存选股结果 CSV（分析场景一般设为 False）
        """
        selector = self.selector_cls()

        all_days = pd.date_range(self.start_date, self.end_date, freq=freq)
        records = []

        for day in all_days:
            day_str_iso = day.strftime("%Y-%m-%d")

            print(f"\n=== 运行 {selector.strategy_name} 选股日 {day_str_iso} ===")
            try:
                # ⭐ 关键改动：这里直接拿 run 返回的 DF，不依赖 CSV
                df_daily = selector.run(date=day, save=save_selection)
                if df_daily is None:
                    cnt = 0
                else:
                    cnt = len(df_daily)
            except Exception as e:
                print(f"⚠️ 选股运行失败 {day_str_iso}: {e}")
                cnt = 0

            records.append({
                "date": day_str_iso,
                "count": cnt,
            })

        # 汇总为 DataFrame 以下部分保持不变
        df_stats = pd.DataFrame(records)
        df_stats["date"] = pd.to_datetime(df_stats["date"])
        df_stats.set_index("date", inplace=True)

        start_str = self.start_date.strftime("%Y%m%d")
        end_str = self.end_date.strftime("%Y%m%d")
        csv_file = self.analysis_output_dir / f"n_pattern_counts_{start_str}_{end_str}.csv"
        df_stats.to_csv(csv_file, encoding="utf-8-sig")

        print("\n✅ 统计完成")
        print(f"   时间区间: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"   结果文件: {csv_file}")

        print("\n--- 简要统计 ---")
        print("总天数:", len(df_stats))
        print("有选股的天数:", (df_stats["count"] > 0).sum())
        print("平均每天入选数:", df_stats["count"].mean())
        print("最大单日入选数:", df_stats["count"].max())

        if plot:
            self._plot_daily_counts(df_stats, start_str, end_str)
            self._plot_monthly_summary(df_stats, start_str, end_str)

        return df_stats

    # === 以下是可视化辅助函数 ===

    def _plot_daily_counts(self, df_stats, start_str, end_str):
        """
        绘制全区间“每日选股数量”的折线图
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        df_stats["count"].plot(ax=ax)

        ax.set_title(f"NPatternSelector 每日选股数量 ({start_str} ~ {end_str})")
        ax.set_xlabel("日期")
        ax.set_ylabel("入选个数")

        fig.tight_layout()

        png_file = self.analysis_output_dir / f"n_pattern_counts_daily_{start_str}_{end_str}.png"
        fig.savefig(png_file, dpi=150)
        plt.close(fig)

        print(f"📈 每日选股数量折线图已保存: {png_file}")

    def _plot_monthly_summary(self, df_stats, start_str, end_str):
        """
        按月份汇总，画柱状图：
        - x 轴：月份
        - y 轴：该月平均每天入选个数
        你也可以改成 .sum() 看总数量
        """
        # 按月取平均（也可以改成 sum 看总数）
        monthly_avg = df_stats["count"].resample("M").mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        monthly_avg.plot(kind="bar", ax=ax)

        ax.set_title(f"NPatternSelector 月度平均每日选股数量 ({start_str} ~ {end_str})")
        ax.set_xlabel("月份")
        ax.set_ylabel("平均每日入选个数")

        # x 轴标签略多，可以倾斜一点
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

        fig.tight_layout()

        png_file = self.analysis_output_dir / f"n_pattern_counts_monthly_{start_str}_{end_str}.png"
        fig.savefig(png_file, dpi=150)
        plt.close(fig)

        print(f"📊 月度平均每日选股数量柱状图已保存: {png_file}")
