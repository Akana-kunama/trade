import re
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from strategy_pool.selectors.analyzer.base_analyzer import SelectorAnalyzerBase


class NPatternAnalyzer(SelectorAnalyzerBase):
    """
    分析 1板n调 (n=1,2,3)并构造所需数据集

    """

    def __init__(
        self,
        start_date,
        end_date,
        pool_dir,
        market_data_dir,
        analysis_output_dir=None,
        only_trade_days=True,
        missing="skip",
        exchange_calendar="XSHG"
    ):
        self.pool_dir = Path(pool_dir)
        self.market_data_dir = Path(market_data_dir)

        if analysis_output_dir is None:
            analysis_output_dir = self.pool_dir / "analysis"

        super().__init__(
            start_date=start_date,
            end_date=end_date,
            analysis_output_dir=analysis_output_dir,
            code_col="code",
            date_col="select_time",
            exchange_calendar=exchange_calendar
        )

        self.df_selection = self.load_range_selection_data(
            pool_dir=self.pool_dir,
            only_trade_days=only_trade_days,
            missing=missing
        )

        if self.df_selection is None or self.df_selection.empty:
            raise ValueError("❌ 选股数据加载失败或为空")

        # 统一日期格式
        self.df_selection[self.date_col] = pd.to_datetime(
            self.df_selection[self.date_col], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # 只保留 1板1/2/3调
        self.df_selection = self.df_selection[self.df_selection["Pattern"].astype(str).str.contains(r"1板[123]调", na=False)].copy()


        print(f"✅ df_selection loaded: rows={len(self.df_selection)}, patterns={self.df_selection['Pattern'].unique()}")

    # -------------------------
    # code 规范化：2949 -> 002949.SZ
    # -------------------------
    def normalize_code(self, code) -> str:
        if code is None or (isinstance(code, float) and pd.isna(code)):
            return ""
        s = str(code).strip()
        if s.endswith((".SZ", ".SH", ".BJ")):
            return s
        s = s.replace("SZ", "").replace("SH", "").replace("BJ", "").replace(".", "").strip()
        if s.isdigit():
            s6 = s.zfill(6)
            if s6.startswith(("60", "68", "90")):
                return f"{s6}.SH"
            elif s6.startswith(("00", "30", "20")):
                return f"{s6}.SZ"
            elif s6.startswith(("8", "4")):
                return f"{s6}.BJ"
            else:
                return f"{s6}.SZ"
        return s

    # -------------------------
    # parquet 载入：000001.SZ.parquet
    # -------------------------
    @lru_cache(maxsize=512)
    def load_price_df(self, code: str) -> pd.DataFrame | None:
        code_std = self.normalize_code(code)
        if not code_std:
            return None

        fp = self.market_data_dir / f"{code_std}.parquet"
        if not fp.exists():
            print(f"⚠️ 找不到日线 parquet：{fp}（raw={code} -> std={code_std}）")
            return None

        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()
        df.columns = [c.lower() for c in df.columns]
        return df

    # -------------------------
    # 未来窗口：按交易日K线根数切片
    # -------------------------
    def get_forward_window(self, price_df: pd.DataFrame, select_time, n: int) -> pd.DataFrame:
        t0 = pd.to_datetime(select_time)
        idx = price_df.index
        pos = idx.searchsorted(t0)
        if pos >= len(idx):
            return price_df.iloc[0:0]
        return price_df.iloc[pos: pos + n]
    
    # -------------------------
    # 过去窗口：按交易日K线根数切片
    # -------------------------
    def get_backward_window_excl_today(self, price_df: pd.DataFrame, select_time, n: int) -> pd.DataFrame:
        """
        从 select_time 向过去取 n 个交易日（不含当天）
        """
        t0 = pd.to_datetime(select_time)
        idx = price_df.index

        pos = idx.searchsorted(t0, side="right") - 1
        if pos <= 0:
            return price_df.iloc[0:0]

        end = pos            # 不含 pos
        start = max(0, end - n)
        return price_df.iloc[start:end]

    # -------------------------
    # 解析 n：1板2调 -> 2
    # -------------------------
    @staticmethod
    def parse_n_from_pattern(pat: str) -> int:
        m = re.search(r"1板(\d)调", str(pat))
        return int(m.group(1)) if m else 0



    # -------------------------
    # 构造数据分析集合
    # -------------------------
    def analyze_data_build(self, window_days: int = 10, past_days: int = 20) -> pd.DataFrame:
        """
        T日收盘后选股，T+1开盘买，T+2收盘卖。
        特征：仅使用 <=T 的信息（含T日OHLCV）。
        标签：使用 T+1/T+2 以及未来 window_days 内极值。
        """

        df = self.df_selection.copy()
        records = []

        for _, row in df.iterrows():
            code = row.get("code")
            t0 = row.get(self.date_col)  # select_time, 'YYYY-MM-DD'

            if pd.isna(code) or pd.isna(t0):
                continue

            px = self.load_price_df(code)
            if px is None or px.empty:
                continue

            px = px.sort_index()
            # 必要列
            need = {"open","high","low","close","volume","amount","limit_up","limit_down"}
            if not need.issubset(px.columns):
                continue

            # ====== 取窗口 ======
            # 未来窗口：从T开始取 T..T+window_days（至少3根：T,T+1,T+2）
            f_w = self.get_forward_window(px, t0, n=window_days + 1)
            if f_w is None or len(f_w) < 3:
                continue

            # 过去窗口：不含当天，取 past_days 根（用于均线/波动等）
            p_w = self.get_backward_window_excl_today(px, t0, n=past_days)
            if p_w is None or len(p_w) < 5:
                # past_days不足也可以跳过或仅用少量特征；这里选择跳过更干净
                continue

            # ====== 明确锚点 ======
            # f_w.iloc[0] = T
            # f_w.iloc[1] = T+1
            # f_w.iloc[2] = T+2
            T = f_w.iloc[0]
            T1 = f_w.iloc[1]
            T2 = f_w.iloc[2]

            open_T, high_T, low_T, close_T = map(float, [T.open, T.high, T.low, T.close])
            vol_T, amt_T = float(T.volume), float(T.amount)
            limit_up_T, limit_down_T = float(T.limit_up), float(T.limit_down)

            open_t1, close_t1 = float(T1.open), float(T1.close)
            open_t2, close_t2 = float(T2.open), float(T2.close)

            # ====== 特征：T日K线结构 ======
            upper_shadow_T = (high_T - max(open_T, close_T)) / close_T if close_T else np.nan
            lower_shadow_T = (min(open_T, close_T) - low_T) / close_T if close_T else np.nan
            body_T = abs(close_T - open_T) / close_T if close_T else np.nan
            range_T = (high_T - low_T) / close_T if close_T else np.nan
            close_pos_T = (close_T - low_T) / (high_T - low_T + 1e-12)

            # ====== 特征：用“工程版 preClose” ======
            # 用过去窗口最后一天 close 作为昨天收盘（不含T）
            pre_close = float(p_w["close"].iloc[-1])
            ret_1d_T = (close_T - pre_close) / pre_close if pre_close else np.nan
            gap_T = (open_T - pre_close) / pre_close if pre_close else np.nan

            # ====== 特征：均线与乖离（全部用过去窗口 + 当天close_T） ======
            def sma(series: pd.Series, n: int) -> float:
                if len(series) < n:
                    return np.nan
                return float(series.iloc[-n:].mean())

            close_hist = p_w["close"]
            vol_hist = p_w["volume"]
            amt_hist = p_w["amount"]

            ma5 = sma(close_hist, 5)
            ma10 = sma(close_hist, 10)
            ma20 = sma(close_hist, 20)

            ma5_bias_T = (close_T - ma5) / ma5 if ma5 and ma5 != 0 else np.nan
            ma10_bias_T = (close_T - ma10) / ma10 if ma10 and ma10 != 0 else np.nan
            ma20_bias_T = (close_T - ma20) / ma20 if ma20 and ma20 != 0 else np.nan

            # 多日收益（用过去窗口 close）
            def ret_n(close_series: pd.Series, n: int) -> float:
                # 计算：close_T vs n日前close（p_w里最后一个是T-1）
                # 取 n 日前：p_w.iloc[-n] 对应 T-n
                if len(close_series) < n:
                    return np.nan
                base = float(close_series.iloc[-n])
                return (close_T - base) / base if base else np.nan

            ret_3d_T = ret_n(close_hist, 3)
            ret_5d_T = ret_n(close_hist, 5)
            ret_10d_T = ret_n(close_hist, 10)

            # ====== 特征：量能（当日 vs 过去均值） ======
            vma5 = sma(vol_hist, 5)
            ama5 = sma(amt_hist, 5)
            vol_ratio_5_T = (vol_T / vma5) if vma5 and vma5 != 0 else np.nan
            amt_ratio_5_T = (amt_T / ama5) if ama5 and ama5 != 0 else np.nan
            amt_ma5_T = ama5

            # ====== 特征：波动/ATR（用过去窗口） ======
            # 用过去窗口计算日收益率波动
            rets = close_hist.pct_change(fill_method=None).dropna()
            volat_5_T = float(rets.iloc[-5:].std()) if len(rets) >= 5 else np.nan
            volat_10_T = float(rets.iloc[-10:].std()) if len(rets) >= 10 else np.nan

            # ATR: TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
            # 用过去窗口自身构造TR（p_w里不含T）
            prev_close_hist = close_hist.shift(1)
            tr = pd.concat([
                (p_w["high"] - p_w["low"]),
                (p_w["high"] - prev_close_hist).abs(),
                (p_w["low"] - prev_close_hist).abs(),
            ], axis=1).max(axis=1).dropna()

            atr_5_T = float(tr.iloc[-5:].mean()) if len(tr) >= 5 else np.nan
            atr_10_T = float(tr.iloc[-10:].mean()) if len(tr) >= 10 else np.nan
            # 归一化ATR（更适合树模型）
            atr_5n_T = (atr_5_T / close_T) if close_T else np.nan
            atr_10n_T = (atr_10_T / close_T) if close_T else np.nan

            # ====== 特征：涨跌停约束（T日状态） ======
            is_limit_up_T = int(np.isclose(close_T, limit_up_T, atol=1e-6) or close_T >= limit_up_T - 1e-6)
            is_limit_down_T = int(np.isclose(close_T, limit_down_T, atol=1e-6) or close_T <= limit_down_T + 1e-6)
            close_to_limit_up_pct_T = (limit_up_T - close_T) / close_T if close_T else np.nan

            # ====== 标签/统计：未来极值（从T+1开始（含）） ======
            future = f_w.iloc[1:]  # T+1..T+window_days
            future_close = future["close"]
            future_open = future["open"]
            future_high = future["high"]
            future_low = future["low"]

            # close/open 极值
            max_close = float(future_close.max())
            min_close = float(future_close.min())
            max_open = float(future_open.max())
            min_open = float(future_open.min())

            days_to_max_close = int(future_close.values.argmax()) + 1
            days_to_min_close = int(future_close.values.argmin()) + 1
            days_to_max_open = int(future_open.values.argmax()) + 1
            days_to_min_open = int(future_open.values.argmin()) + 1

            max_ret_pct_close = (max_close - close_t1) / close_t1 * 100.0
            min_ret_pct_close = (min_close - close_t1) / close_t1 * 100.0
            max_ret_pct_open = (max_open - open_t1) / open_t1 * 100.0
            min_ret_pct_open = (min_open - open_t1) / open_t1 * 100.0

            # high/low 极值：MFE/MAE（从买入 open_t1）
            future_high_max = float(future_high.max())
            future_low_min = float(future_low.min())
            days_to_high_max = int(future_high.values.argmax()) + 1
            days_to_low_min = int(future_low.values.argmin()) + 1

            mfe_t1o_pct = (future_high_max - open_t1) / open_t1 * 100.0
            mae_t1o_pct = (future_low_min - open_t1) / open_t1 * 100.0

            # 速度比
            max_gain_ratio_close = max_ret_pct_close / days_to_max_close if days_to_max_close > 0 else np.nan
            min_gain_ratio_close = min_ret_pct_close / days_to_min_close if days_to_min_close > 0 else np.nan
            max_gain_ratio_open = max_ret_pct_open / days_to_max_open if days_to_max_open > 0 else np.nan
            min_gain_ratio_open = min_ret_pct_open / days_to_min_open if days_to_min_open > 0 else np.nan

            # ====== 交易规则标签：T+1买 T+2卖 ======
            ret_t1o_t2c_pct = (close_t2 - open_t1) / open_t1 * 100.0
            win_t1o_t2c = int(ret_t1o_t2c_pct > 0.3)

            records.append({
                "code": self.normalize_code(code),
                "select_time": t0,

                # 来自选股CSV的字段
                "Pattern": row.get("Pattern"),
                "n_adjust": row.get("n_adjust"),
                "board_limit_to_select_close_adj_pct": row.get("board_limit_to_select_close_adj_pct"),
                "avg_limit_to_close_adj_per": row.get("avg_limit_to_close_adj_per"),
                "board_close_to_select_close_adj_pct": row.get("board_close_to_select_close_adj_pct"),
                "avg_close_to_close_adj_per": row.get("avg_close_to_close_adj_per"),

                # ===== 特征（T日）=====
                "upper_shadow_T": upper_shadow_T,
                "lower_shadow_T": lower_shadow_T,
                "body_T": body_T,
                "range_T": range_T,
                "close_pos_T": close_pos_T,
                "ret_1d_T": ret_1d_T,
                "gap_T": gap_T,

                "ma5_bias_T": ma5_bias_T,
                "ma10_bias_T": ma10_bias_T,
                "ma20_bias_T": ma20_bias_T,
                "ret_3d_T": ret_3d_T,
                "ret_5d_T": ret_5d_T,
                "ret_10d_T": ret_10d_T,

                "vol_ratio_5_T": vol_ratio_5_T,
                "amt_ratio_5_T": amt_ratio_5_T,
                "amt_ma5_T": amt_ma5_T,

                "volatility_5_T": volat_5_T,
                "volatility_10_T": volat_10_T,
                "atr_5n_T": atr_5n_T,
                "atr_10n_T": atr_10n_T,

                "is_limit_up_T": is_limit_up_T,
                "is_limit_down_T": is_limit_down_T,
                "close_to_limit_up_pct_T": close_to_limit_up_pct_T,

                # ===== 路径点（T+1/T+2）=====
                "open_t1": open_t1,
                "open_t2": open_t2,
                "close_t1": close_t1,
                "close_t2": close_t2,

                # ===== 未来极值统计 =====
                "future_close_max": max_close,
                "future_close_min": min_close,
                "future_open_max": max_open,
                "future_open_min": min_open,

                "future_high_max": future_high_max,
                "future_low_min": future_low_min,
                "days_to_high_max": days_to_high_max,
                "days_to_low_min": days_to_low_min,
                "mfe_t1o_pct": mfe_t1o_pct,
                "mae_t1o_pct": mae_t1o_pct,

                "days_to_max_close": days_to_max_close,
                "days_to_min_close": days_to_min_close,
                "max_ret_pct_close": max_ret_pct_close,
                "min_ret_pct_close": min_ret_pct_close,
                "max_gain_ratio_close": max_gain_ratio_close,
                "min_gain_ratio_close": min_gain_ratio_close,

                "days_to_max_open": days_to_max_open,
                "days_to_min_open": days_to_min_open,
                "max_ret_pct_open": max_ret_pct_open,
                "min_ret_pct_open": min_ret_pct_open,
                "max_gain_ratio_open": max_gain_ratio_open,
                "min_gain_ratio_open": min_gain_ratio_open,

                # ===== 交易标签（核心）=====
                "ret_t1o_t2c_pct": ret_t1o_t2c_pct,
                "win_t1o_t2c": win_t1o_t2c,
            })

        out_df = pd.DataFrame(records)
        out_csv = self.analysis_output_dir / "extrema_speed_analysis.csv"
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"✅ 明细已保存：{out_csv}")
        return out_df
        
    


    # -------------------------
    # 散点图：两指标 vs 极值速度
    # -------------------------
    def plot_scatter_by_n(self, df: pd.DataFrame, x: str, y: str, title: str, filename: str):
        if df is None or df.empty:
            print("⚠️ df为空，跳过绘图")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for n in [1, 2, 3]:
            sub = df[df["n_adjust"] == n].copy()
            if sub.empty:
                continue

            # ✅ 清洗：把 pd.NA -> np.nan，并过滤非有限值
            sub[x] = pd.to_numeric(sub[x], errors="coerce")
            sub[y] = pd.to_numeric(sub[y], errors="coerce")
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])

            if sub.empty:
                continue

            ax.scatter(sub[x].values.astype(float), sub[y].values.astype(float),
                    label=f"1板{n}调", alpha=0.75)

        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        fig.tight_layout()

        out = self.analysis_output_dir / filename
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"📌 已保存：{out}")

    def run_scatter_suite(self, days_forward=10):
        df_ext = self.analyze_data_build(days_forward=days_forward)
        if df_ext.empty:
            print("⚠️ 没有得到 extrema 结果")
            return df_ext

        # x：你要的两个指标
        x1 = "board_limit_to_select_close_adj_pct"
        x2 = "avg_limit_to_close_adj_per"
        x3 = "board_close_to_select_close_adj_pct"
        x4 = "avg_close_to_close_adj_per"

        # y：极值/(极值日-选股日) —— 用收益%/交易日数（速度）
        y1_max = "max_gain_ratio_close"
        y1_min = "min_gain_ratio_close"
        y2_max = "max_gain_ratio_open"
        y2_min = "min_gain_ratio_open"

        self.plot_scatter_by_n(df_ext, x1, y1_max,
            title=f"{x1} vs {y1_max} (extrema, forward={days_forward})",
            filename=f"{x1}_vs_{y1_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x1, y1_min,
            title=f"{x1} vs {y1_min} (extrema, forward={days_forward})",
            filename=f"{x1}_vs_{y1_min}.png"
        )
        self.plot_scatter_by_n(df_ext, x1, y2_max,
            title=f"{x1} vs {y2_max} (extrema, forward={days_forward})",
            filename=f"{x1}_vs_{y2_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x1, y2_min,
            title=f"{x1} vs {y2_min} (extrema, forward={days_forward})",
            filename=f"{x1}_vs_{y2_max}.png"
        )



        self.plot_scatter_by_n(df_ext, x2, y1_max,
            title=f"{x2} vs {y1_max} (extrema, forward={days_forward})",
            filename=f"{x2}_vs_{y1_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x2, y1_min,
            title=f"{x2} vs {y1_min} (extrema, forward={days_forward})",
            filename=f"{x2}_vs_{y1_min}.png"
        )
        self.plot_scatter_by_n(df_ext, x2, y2_max,
            title=f"{x2} vs {y2_max} (extrema, forward={days_forward})",
            filename=f"{x2}_vs_{y2_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x2, y2_min,
            title=f"{x2} vs {y2_min} (extrema, forward={days_forward})",
            filename=f"{x2}_vs_{y2_min}.png"
        )


        self.plot_scatter_by_n(df_ext, x3, y1_max,
            title=f"{x3} vs {y1_max} (extrema, forward={days_forward})",
            filename=f"{x3}_vs_{y1_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x3, y1_min,
            title=f"{x3} vs {y1_min} (extrema, forward={days_forward})",
            filename=f"{x3}_vs_{y1_min}.png"
        )
        self.plot_scatter_by_n(df_ext, x3, y2_max,
            title=f"{x3} vs {y2_max} (extrema, forward={days_forward})",
            filename=f"{x3}_vs_{y2_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x3, y2_min,
            title=f"{x3} vs {y2_min} (extrema, forward={days_forward})",
            filename=f"{x3}_vs_{y2_min}.png"
        )



        self.plot_scatter_by_n(df_ext, x4, y1_max,
            title=f"{x4} vs {y1_max} (extrema, forward={days_forward})",
            filename=f"{x4}_vs_{y1_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x4, y1_min,
            title=f"{x4} vs {y1_min} (extrema, forward={days_forward})",
            filename=f"{x4}_vs_{y1_min}.png"
        )
        self.plot_scatter_by_n(df_ext, x4, y2_max,
            title=f"{x4} vs {y2_max} (extrema, forward={days_forward})",
            filename=f"{x4}_vs_{y2_max}.png"
        )
        self.plot_scatter_by_n(df_ext, x4, y2_min,
            title=f"{x4} vs {y2_min} (extrema, forward={days_forward})",
            filename=f"{x4}_vs_{y2_min}.png"
        )




        return df_ext
