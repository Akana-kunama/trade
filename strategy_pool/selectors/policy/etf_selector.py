# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas_market_calendars as mcal

from strategy_pool.selectors.policy.base_selector import SelectorBase


class ETFWeeklySelector(SelectorBase):
    """
    周频 ETF Selector（只做信号）

    运行规则（按你最新需求）：
    - 选股日 = 每周倒数第二个交易日（因为周最后交易日要执行交易）
    - 回撤条件：选股周(k=0)、前1周(k=1)、前2周(k=2) 三周都为“温和下跌”
      其中每周收益 r 满足：pullback_max_drop <= r < 0

    输出列：
      code, select_time(选股日), trade_time(交易日=本周最后交易日), behave
    """

    def __init__(
        self,
        strategy_name: str,
        parquet_dir: str,

        # universe 输入（二选一）
        universe_csv: str | None = None,
        universe_codes: list[str] | None = None,
        universe_code_col: str = "code",

        # 1) 交易历史
        min_trading_days: int = 252,

        # 2) 流动性
        liq_lookback_days: int = 60,
        min_avg_amount: float = 5e7,

        # 3) 周涨幅门槛
        weekly_lookback_weeks: int = 26,
        min_weekly_gain: float = 0.05,

        # 4) 结构：均线上升 + 连续回撤（3周）
        ma_short: int = 2,
        ma_mid: int = 4,
        slope_weeks: int = 1,
        pullback_weeks: tuple[int, ...] = (0, 1, 2),  # ✅ 固定三周：本周/前1/前2
        pullback_max_drop: float = -0.04,

        # 可选增强：近8周累计涨幅>0
        require_recent_8w_positive: bool = True,

        calendar_name: str = "XSHG",
    ):
        super().__init__(strategy_name=strategy_name)

        self.parquet_dir = Path(parquet_dir)
        self.universe_codes = self._load_universe(universe_csv, universe_codes, universe_code_col)

        # 参数
        self.min_trading_days = int(min_trading_days)
        self.liq_lookback_days = int(liq_lookback_days)
        self.min_avg_amount = float(min_avg_amount)

        self.weekly_lookback_weeks = int(weekly_lookback_weeks)
        self.min_weekly_gain = float(min_weekly_gain)

        self.ma_short = int(ma_short)
        self.ma_mid = int(ma_mid)
        self.slope_weeks = int(slope_weeks)

        self.pullback_weeks = tuple(pullback_weeks)
        self.pullback_max_drop = float(pullback_max_drop)

        self.require_recent_8w_positive = bool(require_recent_8w_positive)

        self.cal = mcal.get_calendar(calendar_name)

    # -----------------------
    # Universe
    # -----------------------
    @staticmethod
    def _load_universe(universe_csv, universe_codes, code_col: str) -> list[str]:
        if universe_codes and len(universe_codes) > 0:
            return [str(x) for x in universe_codes]

        if universe_csv:
            df = pd.read_csv(universe_csv)
            if code_col not in df.columns:
                raise ValueError(f"universe_csv 缺少列 {code_col}: {universe_csv}")
            codes = df[code_col].dropna().astype(str).unique().tolist()
            if not codes:
                raise ValueError(f"universe_csv 读取到的 code 为空: {universe_csv}")
            return codes

        raise ValueError("请提供 universe_csv 或 universe_codes（二选一）")

    # -----------------------
    # 交易日历
    # -----------------------
    def _valid_days(self, start, end) -> pd.DatetimeIndex:
        days = self.cal.valid_days(pd.to_datetime(start), pd.to_datetime(end))
        return pd.DatetimeIndex(days.tz_convert(None)).normalize()

    def last_trading_day_on_or_before(self, date_like) -> pd.Timestamp:
        d = pd.to_datetime(date_like).normalize()
        back = self._valid_days(d - pd.Timedelta(days=40), d)
        if len(back) == 0:
            raise RuntimeError("无法找到 <= date 的交易日，请检查日期范围/日历。")
        return back.max()

    def week_last_trading_day_of(self, any_trading_day: pd.Timestamp) -> pd.Timestamp:
        d = pd.to_datetime(any_trading_day).normalize()
        week_start = d - pd.Timedelta(days=d.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        days = self._valid_days(week_start, week_end)
        return days.max() if len(days) else d

    def week_penultimate_trading_day_of(self, any_trading_day: pd.Timestamp) -> pd.Timestamp:
        """
        ✅ 周倒数第二交易日（选股日）
        - 若该周只有1个交易日，则退化为周最后交易日
        """
        d = pd.to_datetime(any_trading_day).normalize()
        week_start = d - pd.Timedelta(days=d.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        days = self._valid_days(week_start, week_end)
        if len(days) == 0:
            return d
        days = pd.DatetimeIndex(days).sort_values()
        if len(days) == 1:
            return days[-1]
        return days[-2]

    def is_selection_day(self, date_like) -> tuple[bool, pd.Timestamp]:
        """
        ✅ 选股日触发：
        - 取 <= date 的最近交易日 last_td
        - 如果 last_td == 本周倒数第二交易日，则执行
        """
        last_td = self.last_trading_day_on_or_before(date_like)
        sel_td = self.week_penultimate_trading_day_of(last_td)
        return (last_td == sel_td), last_td

    def iter_week_selection_days(self, start, end) -> list[pd.Timestamp]:
        """
        枚举 [start, end] 内每周“倒数第二交易日”（选股日）
        """
        start = pd.to_datetime(start).normalize()
        end = pd.to_datetime(end).normalize()
        tds = self._valid_days(start, end)
        if len(tds) == 0:
            return []

        # 每个交易日映射到周一
        week_start = (tds - pd.to_timedelta(tds.weekday, unit="D")).normalize()
        tmp = pd.DataFrame({"day": tds, "week": week_start})

        out = []
        for _, g in tmp.groupby("week"):
            days = pd.DatetimeIndex(g["day"]).sort_values()
            if len(days) == 1:
                out.append(days[-1])
            else:
                out.append(days[-2])
        return [pd.to_datetime(x).normalize() for x in out]

    # -----------------------
    # 读取 ETF parquet
    # -----------------------
    def load_etf_df(self, code: str) -> pd.DataFrame | None:
        candidates = [
            self.parquet_dir / f"{code}.parquet",
            self.parquet_dir / f"{code.replace('.', '_')}.parquet",
            self.parquet_dir / f"{code.replace('SH', 'XSHG').replace('SZ', 'XSHE')}.parquet",
            self.parquet_dir / f"{code.replace('SH', 'XSHG').replace('SZ', 'XSHE').replace('.', '_')}.parquet",
        ]
        for p in candidates:
            if p.exists():
                df = pd.read_parquet(p)
                df = df.sort_index()
                df.index = pd.to_datetime(df.index).normalize()
                return df
        return None

    # -----------------------
    # 周频构造（按交易日历分周）
    # -----------------------
    def to_weekly_by_calendar(self, df_daily: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame | None:
        """
        ✅ 周频构造（截至 end_date）
        - 每周只保留一个点：该周“截至 end_date 的最后交易日”
        * 对于完整周：就是周最后交易日（trade_td）
        * 对于选股周（end_date=周倒数第二交易日）：就是 select_td（而不是 trade_td）
        - close：取该周末点 close
        - amount：该周内累计成交额（同样截至该周末点）
        返回：
        index = week_end(每周末点，最后一个点==end_date所在周的末点)
        columns = close, amount
        """
        end_date = pd.to_datetime(end_date).normalize()

        # 截至 end_date 的日线
        df = df_daily.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df[df.index <= end_date]
        if df.empty:
            return None

        # 有效日（close 有值且成交额>0）
        valid = df["close"].notna() & df["amount"].fillna(0).gt(0)
        df = df[valid]
        if df.empty:
            return None

        # 交易日集合（<= end_date）
        td = self._valid_days(df.index.min(), end_date).intersection(df.index)
        if len(td) == 0:
            return None

        # 每个交易日对应周一
        week_start = (td - pd.to_timedelta(td.weekday, unit="D")).normalize()
        tmp = pd.DataFrame({"day": td, "week": week_start})

        # ✅ 关键：截至 end_date 的“每周末点”
        # 对本周而言：week_end 就是 select_td（因为 trade_td > end_date 不在 td 内）
        week_end = tmp.groupby("week")["day"].max().sort_values()
        week_end = pd.DatetimeIndex(week_end.values)

        # 周收盘：取 week_end 对应 close
        w_close = df["close"].reindex(week_end)

        # 周成交额：按周累计（截至 week_end）
        aligned_amt = df["amount"].reindex(td).fillna(0)
        w_amt = aligned_amt.groupby(week_start).sum()
        w_amt.index = pd.to_datetime(w_amt.index)

        # 拼表：index=week_end；amount 按 week_start 对齐（week_end 的周一）
        week_start_for_end = (week_end - pd.to_timedelta(week_end.weekday, unit="D")).normalize()
        out = pd.DataFrame(
            {
                "close": w_close.values,
                "amount": w_amt.reindex(week_start_for_end).values,
            },
            index=week_end,
        )

        # 清理
        out = out[out["close"].notna()]
        return out

    # -----------------------
    # 单只 ETF 判定
    # -----------------------
    def check_one(self, code: str, as_of: pd.Timestamp) -> tuple[bool, str, dict]:
        df = self.load_etf_df(code)
        if df is None or df.empty:
            return False, "SKIP_DATA_NOT_FOUND", {}

        df = df[df.index <= as_of]
        if df.empty:
            return False, "SKIP_NO_DATA_BEFORE_ASOF", {}

        # 1) 至少一年有效交易日（按 close & amount）
        valid = df["close"].notna() & df["amount"].fillna(0).gt(0)
        dfv = df[valid]
        if len(dfv) < self.min_trading_days:
            return False, "SKIP_NOT_ENOUGH_TRADING_DAYS", {"trading_days": int(len(dfv))}

        # 2) 流动性（近 N 交易日均额）
        df_liq = dfv.iloc[-self.liq_lookback_days:] if len(dfv) >= self.liq_lookback_days else dfv
        avg_amount = float(df_liq["amount"].mean()) if len(df_liq) else 0.0
        if avg_amount < self.min_avg_amount:
            return False, "SKIP_LOW_LIQUIDITY", {"avg_amount": avg_amount}

        # 周频
        w = self.to_weekly_by_calendar(df, as_of)
        if w is None or w.empty:
            return False, "SKIP_WEEKLY_EMPTY", {}

        need_weeks = max(self.weekly_lookback_weeks, self.ma_mid + self.slope_weeks + 4, 12)
        if len(w) < need_weeks:
            return False, "SKIP_WEEKLY_NOT_ENOUGH", {"weeks": int(len(w))}

        # 3) 周涨幅门槛：近 N 周最大单周涨幅
        wret = w["close"].pct_change()
        max_weekly_gain = float(wret.iloc[-self.weekly_lookback_weeks:].max())
        if not np.isfinite(max_weekly_gain) or max_weekly_gain < self.min_weekly_gain:
            return False, "SKIP_WEEKLY_GAIN_TOO_LOW", {"max_weekly_gain": max_weekly_gain}

        # 可选：近8周累计涨幅 > 0
        ret_recent_8w = np.nan
        if len(w["close"]) >= 9:
            ret_recent_8w = float(w["close"].iloc[-1] / w["close"].iloc[-9] - 1)
            if self.require_recent_8w_positive and (not np.isfinite(ret_recent_8w) or ret_recent_8w <= 0):
                return False, "SKIP_RECENT_8W_NOT_POSITIVE", {"ret_recent_8w": ret_recent_8w}

        # 4) 均线向上
        w = w.copy()
        w["ma_s"] = w["close"].rolling(self.ma_short).mean()
        w["ma_m"] = w["close"].rolling(self.ma_mid).mean()

        if pd.isna(w["ma_s"].iloc[-1]) or pd.isna(w["ma_s"].iloc[-1 - self.slope_weeks]):
            return False, "SKIP_MA_SHORT_NA", {}
        if pd.isna(w["ma_m"].iloc[-1]) or pd.isna(w["ma_m"].iloc[-1 - self.slope_weeks]):
            return False, "SKIP_MA_MID_NA", {}

        ma_s_slope = float(w["ma_s"].iloc[-1] - w["ma_s"].iloc[-1 - self.slope_weeks])
        ma_m_slope = float(w["ma_m"].iloc[-1] - w["ma_m"].iloc[-1 - self.slope_weeks])
        if not (ma_s_slope > 0 and ma_m_slope > 0):
            return False, "SKIP_TREND_NOT_UP", {"ma_s_slope": ma_s_slope, "ma_m_slope": ma_m_slope}

        # 5) ✅ 三周连续温和下跌：k=0(选股周),1,2 都满足 pullback_max_drop <= r < 0
        need = self.pullback_weeks  # 期望 (0,1,2)
        if len(wret) < (max(need) + 2):
            return False, "SKIP_PULLBACK_NOT_ENOUGH_WEEKS", {"weeks": int(len(w))}

        rets = {}
        for k in need:
            r = float(wret.iloc[-1 - k])
            rets[f"wret_k{k}"] = r
            if not (r < 0 and r >= self.pullback_max_drop):
                return False, "SKIP_PULLBACK_3W_NOT_ALL_DOWN", rets

        return True, "PASS", {
            "avg_amount": avg_amount,
            "max_weekly_gain": max_weekly_gain,
            "ret_recent_8w": ret_recent_8w,
            "ma_s_slope": ma_s_slope,
            "ma_m_slope": ma_m_slope,
            **rets
        }

    # -----------------------
    # 主入口：只在“选股日(倒数第二交易日)”执行
    # 只保存：选中的 ETF（PASS）
    # -----------------------
    def run(self, date=None):
        if date is None:
            date = datetime.now()

        ok, select_td = self.is_selection_day(date)
        trade_td = self.week_last_trading_day_of(select_td)

        # 非选股日：不保存文件（也可以选择保存空文件，看你习惯）
        if not ok:
            df_out = pd.DataFrame(columns=["code", "select_time", "trade_time", "behave"])
            return [], df_out

        picked = []
        for code in self.universe_codes:
            passed, behave, _info = self.check_one(code, select_td)
            if passed:
                picked.append(code)

        # ✅ 只输出 PASS
        rows = [{
            "code": code,
            "select_time": select_td.strftime("%Y-%m-%d"),
            "trade_time": trade_td.strftime("%Y-%m-%d"),
            "behave": "PASS"
        } for code in picked]

        df_out = pd.DataFrame(rows, columns=["code", "select_time", "trade_time", "behave"])

        # ✅ 保存仅包含 PASS 的 csv（即使为空也保存表头，便于后续分析）
        self.save_result(df_out, date_str=select_td.strftime("%Y%m%d"))

        return picked, df_out


    # -----------------------
    # 区间逐周执行：按“选股日(倒数第二交易日)”逐周跑
    # 只保存：选中的 ETF（PASS）
    # -----------------------
    def run_range(self, start, end):
        sel_days = self.iter_week_selection_days(start, end)

        picked_map = {}
        df_list = []

        for select_td in sel_days:
            trade_td = self.week_last_trading_day_of(select_td)

            picked = []
            for code in self.universe_codes:
                passed, behave, _info = self.check_one(code, select_td)
                if passed:
                    picked.append(code)

            # ✅ 只输出 PASS
            rows = [{
                "code": code,
                "select_time": select_td.strftime("%Y-%m-%d"),
                "trade_time": trade_td.strftime("%Y-%m-%d"),
                "behave": "PASS"
            } for code in picked]

            df_out = pd.DataFrame(rows, columns=["code", "select_time", "trade_time", "behave"])

            # ✅ 每周都保存（即使空表也保存，确保 YYYYMMDD.csv 连续可追溯）
            self.save_result(df_out, date_str=select_td.strftime("%Y%m%d"))

            picked_map[select_td.strftime("%Y-%m-%d")] = picked
            df_list.append(df_out.assign(select_time=select_td.strftime("%Y-%m-%d")))  # 可选：防止空表缺字段

        df_all = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(
            columns=["code", "select_time", "trade_time", "behave"]
        )
        return picked_map, df_all
