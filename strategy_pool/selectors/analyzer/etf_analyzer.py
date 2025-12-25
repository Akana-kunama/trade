# -*- coding: utf-8 -*-
"""
ETFWeeklySelectorAnalyzer
- 读取 ETFWeeklySelector 输出（每周倒数第二个交易日选，周最后交易日买入；下周周末卖出）
- 构造交易标签：buy=trade_td(open), sell=next_week_last_td(close)
- 构造特征：默认用 t0 = select_td（严格对齐“选股日收盘后可见信息”）
- Pointwise 模型（分类/回归打分）+ TopK（K=1/2/3）对比报告：收益/回撤/卡玛
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from strategy_pool.selectors.analyzer.base_analyzer import SelectorAnalyzerBase


class ETFWeeklySelectorAnalyzer(SelectorAnalyzerBase):
    """
    Pointwise Analyzer for ETFWeeklySelector outputs.

    Pipeline:
      1) load selection csvs (pool_dir/YYYYMMDD.csv) -> df_selection
      2) compute_event_trades(df_selection) -> df_trades (TRADE_OK with net_ret)
      3) build_event_feature_table(df_trades_ok) -> df_event (features @ t0=select_td)
      4) train pointwise model (LGBMClassifier/LGBMRegressor)
      5) score test -> pick TopK per week -> evaluate (weekly portfolio returns)
      6) auto K=1/2/3 compare: CAGR / MaxDD / Calmar (+ mean/win_rate)
    """

    def __init__(
        self,
        start_date,
        end_date,
        pool_dir: str | Path,
        parquet_dir: str | Path,
        analysis_output_dir=None,
        code_col="code",
        date_col="select_time",
        buy_date_col="trade_time",
        behave_col="behave",
        pass_value="PASS",
        open_col="open",
        close_col="close",
        exchange_calendar="XSHG",
        fee_rate=0.0,
        extend_days: int = 14,
    ):
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            analysis_output_dir=analysis_output_dir,
            code_col=code_col,
            date_col=date_col,
            exchange_calendar=exchange_calendar,
        )
        self.pool_dir = Path(pool_dir)
        self.parquet_dir = Path(parquet_dir)

        self.buy_date_col = buy_date_col
        self.behave_col = behave_col
        self.pass_value = pass_value

        self.open_col = open_col
        self.close_col = close_col
        self.fee_rate = float(fee_rate)

        self.extend_days = int(extend_days)
        self._daily_cache: Dict[str, Optional[pd.DataFrame]] = {}

        # ✅ cache for trade days
        self._tds_cache: Optional[pd.DatetimeIndex] = None

    # ---------------------------
    # Trade calendar (cached + extended)
    # ---------------------------
    def _trade_days_index(self) -> pd.DatetimeIndex:
        if self._tds_cache is not None:
            return self._tds_cache

        start = pd.to_datetime(self.start_date).normalize()
        end_plus = pd.to_datetime(self.end_date).normalize() + pd.Timedelta(days=self.extend_days)
        try:
            import pandas_market_calendars as mcal
            cal = mcal.get_calendar(self.exchange_calendar)
            trade_days = cal.valid_days(start, end_plus)
            tds = pd.DatetimeIndex(pd.to_datetime(trade_days).tz_localize(None)).normalize().sort_values()
        except Exception as e:
            print(f"⚠️ calendar fallback to B days: {e}")
            tds = pd.date_range(start, end_plus, freq="B")

        self._tds_cache = tds
        return tds

    def _last_trade_day_of_week(self, d: pd.Timestamp) -> pd.Timestamp:
        d = pd.to_datetime(d).normalize()
        week_start = d - pd.Timedelta(days=d.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        tds = self._trade_days_index()
        days = tds[(tds >= week_start) & (tds <= week_end)]
        return days.max() if len(days) else d

    def _next_week_last_trade_day(self, d: pd.Timestamp) -> pd.Timestamp:
        d = pd.to_datetime(d).normalize()
        return self._last_trade_day_of_week(d + pd.Timedelta(days=7))

    def _prev_trade_day(self, d: pd.Timestamp) -> Optional[pd.Timestamp]:
        d = pd.to_datetime(d).normalize()
        tds = self._trade_days_index()
        pos = tds.searchsorted(d)
        if pos <= 0:
            return None
        return pd.Timestamp(tds[pos - 1]).normalize()

    # ---------------------------
    # Parquet loader (cached)
    # ---------------------------
    def load_etf_daily(self, code: str) -> Optional[pd.DataFrame]:
        code = str(code)
        if code in self._daily_cache:
            return self._daily_cache[code]

        candidates = [
            self.parquet_dir / f"{code}.parquet",
            self.parquet_dir / f"{code.replace('.', '_')}.parquet",
            self.parquet_dir / f"{code.replace('SH', 'XSHG').replace('SZ', 'XSHE')}.parquet",
            self.parquet_dir / f"{code.replace('SH', 'XSHG').replace('SZ', 'XSHE').replace('.', '_')}.parquet",
        ]
        df = None
        for p in candidates:
            if p.exists():
                df = pd.read_parquet(p)
                break
        if df is None or df.empty:
            self._daily_cache[code] = None
            return None

        df = df.sort_index()
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="last")]
        self._daily_cache[code] = df
        return df

    def _get_px(self, df: pd.DataFrame, dt: pd.Timestamp, col: str) -> Optional[float]:
        dt = pd.to_datetime(dt).normalize()
        if df is None or df.empty or dt not in df.index or col not in df.columns:
            return None
        v = df.at[dt, col]
        try:
            v = float(v)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    def _align_to_prev_available(self, daily: pd.DataFrame, t: pd.Timestamp) -> Optional[pd.Timestamp]:
        """在日线里找 <=t 的最近可用日期（防 parquet 缺日）。"""
        if daily is None or daily.empty:
            return None
        t = pd.to_datetime(t).normalize()
        idx = pd.DatetimeIndex(pd.to_datetime(daily.index).normalize())
        m = idx[idx <= t]
        if len(m) == 0:
            return None
        return pd.Timestamp(m.max()).normalize()

    # ---------------------------
    # Event trades (truth)
    # ---------------------------
    def compute_event_trades(self, df_selection: pd.DataFrame, only_pass=True) -> pd.DataFrame:
        """
        buy_td = trade_time（本周最后交易日）
        sell_td = next_week_last_trade_day(buy_td)
        buy_price = buy_td open
        sell_price = sell_td close
        """
        df = df_selection.copy()

        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df[self.buy_date_col] = pd.to_datetime(df[self.buy_date_col], errors="coerce")

        if only_pass and self.behave_col in df.columns:
            df = df[df[self.behave_col] == self.pass_value].copy()

        rows = []
        for _, r in df.iterrows():
            code = str(r.get(self.code_col, ""))

            select_td = pd.to_datetime(r.get(self.date_col, None), errors="coerce")
            buy_td = pd.to_datetime(r.get(self.buy_date_col, None), errors="coerce")

            if pd.isna(select_td):
                rows.append({"code": code, "select_td": pd.NaT, "buy_td": pd.NaT, "sell_td": pd.NaT,
                             "net_ret": np.nan, "behave": "SKIP_BAD_SELECT_DATE"})
                continue
            if pd.isna(buy_td):
                rows.append({"code": code, "select_td": select_td.normalize(), "buy_td": pd.NaT, "sell_td": pd.NaT,
                             "net_ret": np.nan, "behave": "SKIP_BAD_BUY_DATE"})
                continue

            select_td = select_td.normalize()
            buy_td = buy_td.normalize()
            sell_td = self._next_week_last_trade_day(buy_td)

            daily = self.load_etf_daily(code)
            if daily is None:
                rows.append({"code": code, "select_td": select_td, "buy_td": buy_td, "sell_td": sell_td,
                             "net_ret": np.nan, "behave": "SKIP_DATA_NOT_FOUND"})
                continue

            last_avail = pd.to_datetime(daily.index.max()).normalize()
            if sell_td > last_avail:
                rows.append({"code": code, "select_td": select_td, "buy_td": buy_td, "sell_td": sell_td,
                             "net_ret": np.nan, "behave": "SKIP_OUT_OF_DATA_RANGE", "last_avail": last_avail})
                continue

            buy_open = self._get_px(daily, buy_td, self.open_col)
            sell_close = self._get_px(daily, sell_td, self.close_col)

            if buy_open is None:
                rows.append({"code": code, "select_td": select_td, "buy_td": buy_td, "sell_td": sell_td,
                             "net_ret": np.nan, "behave": "SKIP_NO_BUY_OPEN"})
                continue
            if sell_close is None:
                rows.append({"code": code, "select_td": select_td, "buy_td": buy_td, "sell_td": sell_td,
                             "net_ret": np.nan, "behave": "SKIP_NO_SELL_CLOSE"})
                continue

            gross = sell_close / buy_open - 1.0
            net = (1.0 + gross) * (1.0 - self.fee_rate) * (1.0 - self.fee_rate) - 1.0

            rows.append({
                "code": code, "select_td": select_td, "buy_td": buy_td, "sell_td": sell_td,
                "buy_open": buy_open, "sell_close": sell_close,
                "gross_ret": gross, "net_ret": net,
                "behave": "TRADE_OK",
            })

        return pd.DataFrame(rows)

    # ---------------------------
    # Features (NO scoring)
    # ---------------------------
    def _max_drawdown(self, px: pd.Series) -> float:
        px = pd.to_numeric(px, errors="coerce").dropna()
        if len(px) < 2:
            return np.nan
        peak = px.cummax()
        dd = px / peak - 1.0
        return float(dd.min())

    def feature_one_event(self, daily: pd.DataFrame, t0: pd.Timestamp) -> dict:
        """
        Default daily features (all use history <= t0).
        Robust & low-variance.
        """
        if daily is None or daily.empty:
            return {}
        t0 = pd.to_datetime(t0).normalize()
        if t0 not in daily.index:
            # 对齐到 <=t0 的最近可用日
            t0_eff = self._align_to_prev_available(daily, t0)
            if t0_eff is None:
                return {}
            t0 = t0_eff

        hist = daily.loc[:t0].copy()
        if "close" not in hist.columns:
            return {}

        close = pd.to_numeric(hist["close"], errors="coerce").dropna()
        if len(close) < 60:
            return {}

        r1 = close.pct_change()

        # Momentum / trend
        mom_5 = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else np.nan
        mom_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else np.nan

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma20_slope_5 = float(ma20.iloc[-1] - ma20.iloc[-6]) if len(ma20) >= 6 and pd.notna(ma20.iloc[-6]) else np.nan
        ma20_dev = float(close.iloc[-1] / ma20.iloc[-1] - 1.0) if pd.notna(ma20.iloc[-1]) else np.nan
        ma20_over_ma60 = float(ma20.iloc[-1] / ma60.iloc[-1] - 1.0) if pd.notna(ma20.iloc[-1]) and pd.notna(ma60.iloc[-1]) else np.nan

        # Risk
        vol_20 = float(np.nanstd(r1.iloc[-20:].values, ddof=1)) if len(r1.dropna()) >= 21 else np.nan
        dd_10 = self._max_drawdown(close.iloc[-10:])

        # Liquidity (if amount exists)
        amt_ma20 = np.nan
        if "amount" in hist.columns:
            amt = pd.to_numeric(hist["amount"], errors="coerce")
            amt_ma20 = float(np.nanmean(amt.iloc[-20:].values)) if len(amt) >= 20 else float(np.nanmean(amt.values))

        return {
            "mom_5": mom_5,
            "mom_20": mom_20,
            "ma20_slope_5": ma20_slope_5,
            "ma20_dev": ma20_dev,
            "ma20_over_ma60": ma20_over_ma60,
            "vol_20": vol_20,
            "mdd_10": dd_10,
            "amt_ma20": amt_ma20,
        }

    def build_event_feature_table(self, df_trades_ok: pd.DataFrame) -> pd.DataFrame:
        """
        Build per-event features table for model:
          code, select_td, buy_td, net_ret, (features...), y_rel(optional)
        ✅ 特征时点严格用 select_td（选股日收盘后可见）
        """
        rows = []
        for _, r in df_trades_ok.iterrows():
            code = str(r["code"])
            select_td = pd.to_datetime(r["select_td"]).normalize()
            buy_td = pd.to_datetime(r["buy_td"]).normalize()

            # ✅ t0 = select_td（更清晰，严格对齐你的策略）
            t0 = select_td

            daily = self.load_etf_daily(code)
            if daily is None:
                continue

            feats = self.feature_one_event(daily, t0)
            if not feats:
                continue

            sell_td = pd.to_datetime(r["sell_td"]).normalize()

            rows.append({
                "code": code,
                "select_td": select_td,
                "buy_td": buy_td,
                "sell_td": sell_td,
                "t0": t0,
                "net_ret": float(r["net_ret"]),
                **feats
            })

        df_event = pd.DataFrame(rows)
        if df_event.empty:
            return df_event

        # 可选：保留一个 y_rel（不用于 pointwise，但方便你对比）
        df_event["y_rel"] = (
            df_event.groupby("select_td")["net_ret"]
            .rank(method="dense", ascending=True)
            .astype(int) - 1
        )
        return df_event

    # ===========================
    # Pointwise helpers
    # ===========================
    def _get_feat_cols(self, df_event: pd.DataFrame) -> List[str]:
        drop_cols = {"code", "select_td", "buy_td", "sell_td", "t0", "net_ret", "y_rel", "pred_score"}
        return [c for c in df_event.columns if c not in drop_cols]

    def _prep_X(self, df_event: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
        return (
            df_event[feat_cols]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    def _equity_stats_from_weekly_ret(self, weekly_ret: pd.Series) -> dict:
        """
        weekly_ret: index=select_td(datetime), values=weekly portfolio net_ret(float)
        输出：CAGR, MaxDD, Calmar, mean, win_rate ...
        """
        if weekly_ret is None or weekly_ret.empty:
            return {"n_weeks": 0, "note": "empty"}

        r = pd.to_numeric(weekly_ret, errors="coerce").dropna()
        if r.empty:
            return {"n_weeks": 0, "note": "no valid returns"}

        eq = (1.0 + r).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1.0
        mdd = float(dd.min())  # negative

        start_dt = pd.to_datetime(r.index.min()).normalize()
        end_dt = pd.to_datetime(r.index.max()).normalize()
        days = max((end_dt - start_dt).days, 1)
        years = days / 365.25

        total_ret = float(eq.iloc[-1] - 1.0)
        cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
        calmar = float(cagr / abs(mdd)) if np.isfinite(cagr) and np.isfinite(mdd) and mdd < 0 else np.nan

        return {
            "n_weeks": int(r.shape[0]),
            "win_rate": float((r > 0).mean()),
            "mean_weekly": float(r.mean()),
            "median_weekly": float(r.median()),
            "total_ret": total_ret,
            "cagr": cagr,
            "max_drawdown": mdd,
            "calmar": calmar,
            "min_weekly": float(r.min()),
            "max_weekly": float(r.max()),
            "std_weekly": float(r.std(ddof=1)) if r.shape[0] >= 2 else float("nan"),
        }

    def _pick_topk_per_week(self, df_scored: pd.DataFrame, k: int) -> pd.DataFrame:
        return (
            df_scored.sort_values(["select_td", "pred_score"], ascending=[True, False])
            .groupby("select_td")
            .head(int(k))
            .copy()
        )

    def _weekly_portfolio_return(self, df_pick: pd.DataFrame) -> pd.Series:
        if df_pick is None or df_pick.empty:
            return pd.Series(dtype=float)
        return (
            df_pick.groupby("select_td")["net_ret"]
            .mean()
            .sort_index()
        )

    def _purged_time_split(
            self,
            df_event: pd.DataFrame,
            train_end_dt: pd.Timestamp,
            valid_end_dt: Optional[pd.Timestamp] = None,
            purge_weeks: int = 1,
        ):
            """
            Purged split by label horizon:
            - train: sell_td <= train_end_dt
            - valid: (optional) sell_td in (train_end_dt, valid_end_dt] with purge
            - test : select_td > valid_end_dt (or > train_end_dt if no valid)
            purge_weeks: 在切分点前后，剔除若干周的 select_td，减少边界污染
            """
            df = df_event.copy()
            df["select_td"] = pd.to_datetime(df["select_td"]).dt.normalize()
            df["sell_td"]   = pd.to_datetime(df["sell_td"]).dt.normalize()

            # 训练集必须“收益已完全落地”
            train = df[df["sell_td"] <= train_end_dt].copy()

            cut = valid_end_dt if valid_end_dt is not None else train_end_dt

            # purge：剔除靠近 cut 的若干周（按周频，这个很有效）
            if purge_weeks and purge_weeks > 0:
                left  = cut - pd.Timedelta(days=7 * purge_weeks)
                right = cut + pd.Timedelta(days=7 * purge_weeks)
                df = df[~df["select_td"].between(left, right)].copy()

                # purge 之后重新定义 train / test
                train = df[df["sell_td"] <= train_end_dt].copy()

            if valid_end_dt is None:
                valid = None
                test = df[df["select_td"] > train_end_dt].copy()
                test_cut = train_end_dt
            else:
                valid = df[(df["sell_td"] > train_end_dt) & (df["sell_td"] <= valid_end_dt)].copy()
                test = df[df["select_td"] > valid_end_dt].copy()
                test_cut = valid_end_dt

            return train, valid, test, test_cut


    # ===========================
    # Pointwise training
    # ===========================
    def train_lgbm_pointwise(
        self,
        df_event_train: pd.DataFrame,
        df_event_valid: Optional[pd.DataFrame] = None,
        mode: str = "cls",   # "cls" or "reg"
        params: Optional[dict] = None,
    ):
        """
        Pointwise model:
          - cls: 预测 P(win) = P(net_ret > 0)
          - reg: 预测 net_ret
        返回：model, feat_cols
        """
        if df_event_train is None or df_event_train.empty:
            raise RuntimeError("Empty training dataset (df_event_train).")

        feat_cols = self._get_feat_cols(df_event_train)
        X_train = self._prep_X(df_event_train, feat_cols)

        if mode == "cls":
            try:
                from lightgbm import LGBMClassifier
            except Exception as e:
                raise RuntimeError("LightGBM not installed. Please: pip install lightgbm") from e

            y_train = (pd.to_numeric(df_event_train["net_ret"], errors="coerce").fillna(0.0) > 0).astype(int).values

            if params is None:
                # 偏“稳健”的默认：更强正则 + 更大叶子，适合你 PASS 小、噪声大的周频
                params = dict(
                    objective="binary",
                    learning_rate=0.03,
                    n_estimators=200,
                    num_leaves=30,
                    max_depth=4,
                    min_data_in_leaf=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                )

            model = LGBMClassifier(**params)

            if df_event_valid is not None and not df_event_valid.empty:
                X_val = self._prep_X(df_event_valid, feat_cols)
                y_val = (pd.to_numeric(df_event_valid["net_ret"], errors="coerce").fillna(0.0) > 0.02).astype(int).values
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=[
                        # 早停 + 保留最佳迭代
                        __import__("lightgbm").early_stopping(stopping_rounds=100, verbose=False),
                        __import__("lightgbm").log_evaluation(period=0),
                    ],
                )
            else:
                model.fit(X_train, y_train)

            return model, feat_cols

        elif mode == "reg":
            try:
                from lightgbm import LGBMRegressor
            except Exception as e:
                raise RuntimeError("LightGBM not installed. Please: pip install lightgbm") from e

            y_train = pd.to_numeric(df_event_train["net_ret"], errors="coerce").fillna(0.0).values

            if params is None:
                params = dict(
                    objective="regression",
                    learning_rate=0.03,
                    n_estimators=2000,
                    num_leaves=31,
                    max_depth=4,
                    min_data_in_leaf=30,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                )

            model = LGBMRegressor(**params)

            if df_event_valid is not None and not df_event_valid.empty:
                X_val = self._prep_X(df_event_valid, feat_cols)
                y_val = pd.to_numeric(df_event_valid["net_ret"], errors="coerce").fillna(0.0).values
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="l2",
                )
            else:
                model.fit(X_train, y_train)

            return model, feat_cols

        else:
            raise ValueError("mode must be 'cls' or 'reg'")

    # ===========================
    # End-to-end pointwise backtest + K compare
    # ===========================
    def run_pointwise_backtest(
        self,
        only_trade_days=True,
        missing="skip",
        only_pass=True,                 # ✅ 两阶段：只在 PASS 候选里挑
        train_end: str = "2023-12-31",
        valid_end: Optional[str] = None,
        ks=(1, 2, 3),                   # ✅ 自动比较 K=1/2/3
        mode: str = "cls",              # ✅ 默认分类更稳
        save_csv: bool = True,
        tag: str = "lgbm_pointwise",
        params: Optional[dict] = None,  # 可传入覆盖默认模型参数
    ):
        """
        返回：
          df_event: 全事件特征表
          df_test:  测试集含 pred_score
          results:  dict，含 ALL_PASS_EQUAL / TOP1 / TOP2 / TOP3 的收益/回撤/卡玛
          meta:     一些元信息（test开始日期、mode、ks）
        """
        # 1) load selections -> trades -> ok trades
        df_sel = self.load_range_selection_data(self.pool_dir, only_trade_days=only_trade_days, missing=missing)
        df_trades = self.compute_event_trades(df_sel, only_pass=only_pass)
        df_ok = df_trades[df_trades["behave"].eq("TRADE_OK")].copy()
        if df_ok.empty:
            raise RuntimeError("No TRADE_OK trades in range.")

        # 2) build event table
        df_event = self.build_event_feature_table(df_ok)
        if df_event.empty:
            raise RuntimeError("Event feature table is empty (check parquet columns / dates).")

        # 3) time split
        train_end_dt = pd.to_datetime(train_end).normalize()
        valid_end_dt = pd.to_datetime(valid_end).normalize() if valid_end else None

        df_train, df_valid, df_test, test_cut = self._purged_time_split(
            df_event=df_event,
            train_end_dt=train_end_dt,
            valid_end_dt=valid_end_dt,
            purge_weeks=1,   # 你也可以设 0 关闭
        )

        if df_train.empty or df_test.empty:
            raise RuntimeError(f"Train/Test empty after purged split. train_end={train_end}, valid_end={valid_end}.")

        # 4) train model
        model, feat_cols = self.train_lgbm_pointwise(df_train, df_valid, mode=mode, params=params)

        # 5) score test
        df_test = df_test.sort_values(["select_td", "code"]).reset_index(drop=True)
        X_test = self._prep_X(df_test, feat_cols)

        if mode == "cls":
            proba = model.predict_proba(X_test)
            df_test["pred_score"] = proba[:, 1].astype(float)  # P(win)
        else:
            df_test["pred_score"] = model.predict(X_test).astype(float)  # predicted return

        # 6) K compare (weekly portfolio)
        results: Dict[str, dict] = {}

        # baseline：test期每周把“所有 PASS 样本”都等权买入
        weekly_all = self._weekly_portfolio_return(df_test)
        results["ALL_PASS_EQUAL"] = self._equity_stats_from_weekly_ret(weekly_all)

        all_pick_concat = []
        for k in ks:
            df_pick = self._pick_topk_per_week(df_test, int(k))
            weekly_k = self._weekly_portfolio_return(df_pick)
            results[f"TOP{k}"] = self._equity_stats_from_weekly_ret(weekly_k)

            # 附加一些“交易周数/空周数”信息
            total_weeks = int(df_test["select_td"].nunique())
            traded_weeks = int(df_pick["select_td"].nunique())
            empty_weeks = total_weeks - traded_weeks
            results[f"TOP{k}"]["total_weeks"] = total_weeks
            results[f"TOP{k}"]["traded_weeks"] = traded_weeks
            results[f"TOP{k}"]["empty_weeks"] = empty_weeks

            all_pick_concat.append(df_pick.assign(pick_k=int(k)))

            if save_csv:
                self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
                out_pick = self.analysis_output_dir / f"{tag}_top{k}_picks.csv"
                df_pick.to_csv(out_pick, index=False, encoding="utf-8-sig")

        # 7) save csvs
        if save_csv:
            self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
            out_event = self.analysis_output_dir / f"{tag}_event_table.csv"
            out_test  = self.analysis_output_dir / f"{tag}_test_scored.csv"
            out_sum   = self.analysis_output_dir / f"{tag}_K_compare_summary.csv"

            df_event.to_csv(out_event, index=False, encoding="utf-8-sig")
            df_test.to_csv(out_test, index=False, encoding="utf-8-sig")

            sum_rows = [{"scope": name, **stat} for name, stat in results.items()]
            pd.DataFrame(sum_rows).to_csv(out_sum, index=False, encoding="utf-8-sig")

            if all_pick_concat:
                out_allpicks = self.analysis_output_dir / f"{tag}_all_picks_concat.csv"
                pd.concat(all_pick_concat, ignore_index=True).to_csv(out_allpicks, index=False, encoding="utf-8-sig")

            print(f"✅ saved: {out_event}")
            print(f"✅ saved: {out_test}")
            print(f"✅ saved: {out_sum}")

        meta = {
            "test_start_after": str(test_cut.date()),
            "mode": mode,
            "ks": list(ks),
            "only_pass": bool(only_pass),
        }
        return df_event, df_test, results, meta
