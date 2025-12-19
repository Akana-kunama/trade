# -*- coding: utf-8 -*-
"""
Module: selector_analyzer_base.py
Description: 通用选股策略分析基类（SelectorAnalyzerBase） - 增强版
- 支持：按时间段自动载入 pool_storage 下的 YYYYMMDD.csv
- 支持：自动判别交易日（优先 pandas_market_calendars；否则 fallback 工作日）
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SelectorAnalyzerBase:
    """
    通用选股策略分析基类：负责分析已生成的选股文件数据
    """

    def __init__(self, start_date, end_date, analysis_output_dir=None,
                 code_col="code", date_col="select_time",
                 exchange_calendar="XSHG"):
        """
        :param start_date: 起始日期（字符串或 datetime）
        :param end_date:   结束日期（字符串或 datetime）
        :param analysis_output_dir: 输出目录
        :param code_col: 股票代码列名
        :param date_col: 日期列名（统一到 YYYY-MM-DD 字符串）
        :param exchange_calendar: 交易所日历（pandas_market_calendars），A股常用 XSHG
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.analysis_output_dir = Path(analysis_output_dir) if analysis_output_dir else Path("./analysis_output")
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)

        self.code_col = code_col
        self.date_col = date_col
        self.exchange_calendar = exchange_calendar

    # -----------------------------
    # 交易日日历
    # -----------------------------
    def get_trade_days(self) -> pd.DatetimeIndex:
        """
        返回 [start_date, end_date] 区间内的交易日（DatetimeIndex，naive）
        优先使用 pandas_market_calendars；否则退化为工作日（B）
        """
        try:
            import pandas_market_calendars as mcal
            cal = mcal.get_calendar(self.exchange_calendar)
            trade_days = cal.valid_days(self.start_date, self.end_date)
            # 去掉时区，统一为 naive，便于比较
            trade_days = pd.DatetimeIndex(trade_days).tz_localize(None)
            return trade_days
        except Exception as e:
            print(f"⚠️ 未能使用 pandas_market_calendars（{e}），将退化为工作日(B)日历。")
            return pd.date_range(self.start_date, self.end_date, freq="B")

    # -----------------------------
    # 文件扫描 & 载入
    # -----------------------------
    def scan_selection_files(self, pool_dir: str | Path) -> list[Path]:
        """
        扫描 pool_dir 下形如 YYYYMMDD.csv 的文件，并按日期范围过滤
        """
        pool_dir = Path(pool_dir)
        if not pool_dir.exists():
            raise FileNotFoundError(f"找不到目录：{pool_dir}")

        files = []
        for p in pool_dir.glob("*.csv"):
            stem = p.stem
            # 只接受 YYYYMMDD
            if len(stem) == 8 and stem.isdigit():
                dt = pd.to_datetime(stem, format="%Y%m%d", errors="coerce")
                if pd.notna(dt) and self.start_date <= dt <= self.end_date:
                    files.append(p)

        files = sorted(files, key=lambda x: x.stem)
        return files

    def load_range_selection_data(self, pool_dir: str | Path,
                                  only_trade_days: bool = True,
                                  missing: str = "skip") -> pd.DataFrame:
        """
        载入时间段内的选股文件并合并为一个 df_selection

        :param only_trade_days: 是否仅载入交易日文件（推荐 True）
        :param missing: 区间内交易日缺文件时处理方式：
                        - "skip"：跳过缺失日（默认）
                        - "raise"：直接报错
        """
        pool_dir = Path(pool_dir)
        trade_days = self.get_trade_days() if only_trade_days else None

        # 文件列表（按日期范围）
        files = self.scan_selection_files(pool_dir)
        file_map = {p.stem: p for p in files}  # "YYYYMMDD" -> Path

        # 如果只用交易日：按交易日去找对应文件（更严谨）
        target_days = None
        if only_trade_days:
            target_days = [d.strftime("%Y%m%d") for d in trade_days]
        else:
            # 不筛交易日：按扫描到的文件日期为准
            target_days = sorted(file_map.keys())

        df_list = []
        missing_days = []

        for ymd in target_days:
            p = file_map.get(ymd)
            if p is None:
                missing_days.append(ymd)
                continue

            df = pd.read_csv(p)

            # 统一日期列：如果文件内没有 date_col，就用文件名补
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce").dt.strftime("%Y-%m-%d")
            else:
                df[self.date_col] = pd.to_datetime(ymd, format="%Y%m%d").strftime("%Y-%m-%d")

            df_list.append(df)

        if missing_days:
            msg = f"⚠️ 区间内有 {len(missing_days)} 天缺少选股文件（示例：{missing_days[:5]}）"
            if missing == "raise":
                raise FileNotFoundError(msg)
            else:
                print(msg)

        if not df_list:
            raise RuntimeError("没有载入到任何选股数据（检查日期范围/目录/文件命名）。")

        return pd.concat(df_list, ignore_index=True)

    # -----------------------------
    # 统计分析（交易日序列驱动）
    # -----------------------------
    def run_count_per_day(self, df_selection: pd.DataFrame, plot: bool = True,
                          use_trade_days: bool = True) -> pd.DataFrame:
        """
        统计每个日期的选股数量（默认按交易日统计）
        """
        # 确保 date_col 是 YYYY-MM-DD
        if self.date_col in df_selection.columns:
            df_selection[self.date_col] = pd.to_datetime(df_selection[self.date_col], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            raise KeyError(f"df_selection 缺少日期列：{self.date_col}")

        days = self.get_trade_days() if use_trade_days else pd.date_range(self.start_date, self.end_date, freq="B")

        records = []
        for day in days:
            day_str = day.strftime("%Y-%m-%d")
            daily_count = (df_selection[self.date_col] == day_str).sum()
            records.append({"date": day_str, "count": int(daily_count)})

        df_stats = pd.DataFrame(records).set_index("date")

        self.save_stats(df_stats)
        if plot:
            self.plot_stats(df_stats)

        return df_stats

    def save_stats(self, df_stats: pd.DataFrame):
        file_path = self.analysis_output_dir / "stats.csv"
        df_stats.to_csv(file_path, encoding="utf-8-sig")
        print(f"✅ 统计结果已保存: {file_path}")

    def plot_stats(self, df_stats: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(12, 4))
        df_stats["count"].plot(ax=ax)
        ax.set_title("每日选股数量（交易日）")
        ax.set_xlabel("日期")
        ax.set_ylabel("选股数量")
        fig.tight_layout()
        out = self.analysis_output_dir / "daily_counts.png"
        fig.savefig(out, dpi=150)
        print(f"📈 图表已保存：{out}")

    # -----------------------------
    # 一键：载入 + 分析
    # -----------------------------
    def analyze_range(self, pool_dir: str | Path,
                      only_trade_days: bool = True,
                      missing: str = "skip",
                      plot: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        一键完成：载入时间段选股数据 + 输出每日计数统计
        :return: (df_selection, df_stats)
        """
        df_selection = self.load_range_selection_data(pool_dir, only_trade_days=only_trade_days, missing=missing)
        df_stats = self.run_count_per_day(df_selection, plot=plot, use_trade_days=only_trade_days)
        return df_selection, df_stats
