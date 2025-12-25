# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
import pandas_market_calendars as mcal

try:
    from xtquant import xtdata
except Exception:
    xtdata = None


class ETFUniverseBuilder:
    """
    输入：parquet_dir + [start, end]
    输出：
      - codes: list[str]
      - df_pass / df_fail: DataFrame（含 InstrumentName、OpenDate）
    过滤规则（与您之前一致）：
      - close notna 且 amount > 0 视作有效交易日
      - coverage >= min_coverage
      - max_consecutive_missing <= max_consecutive_missing
      - n_valid >= min_valid_days
      - first_valid_date <= start+宽限交易日
    """

    def __init__(
        self,
        parquet_dir: str,
        start: str,
        end: str,
        exchange_calendar: str = "XSHG",

        min_coverage: float = 0.98,
        max_consecutive_missing: int = 10,
        start_grace_trade_days: int = 5,
        min_valid_days: int = 252,

        # xtdata 增强
        enrich_with_xtdata: bool = True,

        # 可选：上市满 N 天再纳入（用 xtdata.OpenDate；没拿到则不强制）
        min_listed_days: int | None = None,

        # 输出
        out_dir: str | None = None,
        out_tag: str = "etf_universe",
    ):
        self.parquet_dir = Path(parquet_dir)
        self.start = pd.to_datetime(start).normalize()
        self.end = pd.to_datetime(end).normalize()
        self.exchange_calendar = exchange_calendar

        self.min_coverage = min_coverage
        self.max_consecutive_missing = max_consecutive_missing
        self.start_grace_trade_days = start_grace_trade_days
        self.min_valid_days = min_valid_days

        self.enrich_with_xtdata = enrich_with_xtdata and (xtdata is not None)
        self.min_listed_days = min_listed_days

        self.out_dir = Path(out_dir) if out_dir else None
        self.out_tag = out_tag

        self._cal = mcal.get_calendar(exchange_calendar)

    # -----------------------
    # 日历
    # -----------------------
    def _trade_days(self, start, end) -> pd.DatetimeIndex:
        days = self._cal.valid_days(pd.to_datetime(start), pd.to_datetime(end))
        return pd.DatetimeIndex(days.tz_convert(None)).normalize()

    # -----------------------
    # parquet -> code
    # -----------------------
    @staticmethod
    def _infer_code(df: pd.DataFrame, fp: Path) -> str:
        if "code" in df.columns and df["code"].notna().any():
            return str(df["code"].dropna().iloc[0])
        return fp.stem.replace("_", ".")

    # -----------------------
    # 缺失段统计
    # -----------------------
    @staticmethod
    def _max_consecutive_true(mask: pd.Series) -> int:
        max_gap = 0
        cur = 0
        for m in mask.values:
            if m:
                cur += 1
                max_gap = max(max_gap, cur)
            else:
                cur = 0
        return int(max_gap)

    # -----------------------
    # 核心指标
    # -----------------------
    def _metrics_with_calendar(self, df: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict:
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()

        aligned = df.reindex(calendar)

        # ✅ 用你定义的有效标准
        valid = aligned["close"].notna() & aligned["amount"].fillna(0).gt(0)

        first_valid_date = valid.idxmax() if valid.any() else pd.NaT
        coverage = float(valid.mean()) if len(valid) else 0.0

        missing = ~valid
        max_gap = self._max_consecutive_true(missing)

        return {
            "first_valid_date": first_valid_date,
            "coverage": coverage,
            "max_consecutive_missing": max_gap,
            "n_days": int(len(valid)),
            "n_valid": int(valid.sum()),
        }

    # -----------------------
    # xtdata 元信息
    # -----------------------
    @staticmethod
    def _safe_xt_detail(code: str) -> dict:
        if xtdata is None:
            return {}
        try:
            d = xtdata.get_instrument_detail(code)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _normalize_opendate(open_date_raw) -> str:
        s = str(open_date_raw).strip() if open_date_raw is not None else ""
        if (not s) or s in {"0", "00000000"}:
            return ""
        if len(s) == 8 and s.isdigit():
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        return s

    @staticmethod
    def _parse_opendate_to_ts(open_date_raw) -> pd.Timestamp | None:
        s = str(open_date_raw).strip() if open_date_raw is not None else ""
        if (not s) or s in {"0", "00000000"}:
            return pd.NaT
        try:
            if len(s) == 8 and s.isdigit():
                return pd.to_datetime(s, format="%Y%m%d")
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    # -----------------------
    # build
    # -----------------------
    def build(self) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        files = sorted(self.parquet_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {self.parquet_dir}")

        calendar = self._trade_days(self.start, self.end)

        rows = []
        for fp in files:
            df = pd.read_parquet(fp)
            code = self._infer_code(df, fp)
            m = self._metrics_with_calendar(df, calendar)
            rows.append({"code": code, **m})

        df_metrics = pd.DataFrame(rows)

        # 起始宽限：允许 start 后若干交易日开始有效
        if len(calendar) > 0:
            grace_idx = min(self.start_grace_trade_days, len(calendar) - 1)
            start_grace_limit = calendar[grace_idx]
        else:
            start_grace_limit = self.start

        passed_mask = (
            df_metrics["first_valid_date"].notna()
            & (df_metrics["first_valid_date"] <= start_grace_limit)
            & (df_metrics["coverage"] >= self.min_coverage)
            & (df_metrics["max_consecutive_missing"] <= self.max_consecutive_missing)
            & (df_metrics["n_valid"] >= self.min_valid_days)
        )

        df_pass = df_metrics[passed_mask].copy()
        df_fail = df_metrics[~passed_mask].copy()

        # ✅ enrich：名称 + 上市日
        if self.enrich_with_xtdata:
            meta_rows = []
            for code in df_pass["code"].astype(str).unique().tolist():
                d = self._safe_xt_detail(code)
                meta_rows.append({
                    "code": code,
                    "InstrumentName": str(d.get("InstrumentName", "")) if d else "",
                    "OpenDate": self._normalize_opendate(d.get("OpenDate")) if d else "",
                    "OpenDate_raw": str(d.get("OpenDate", "")) if d else "",
                    "ExchangeID": str(d.get("ExchangeID", "")) if d else "",
                    "InstrumentID": str(d.get("InstrumentID", "")) if d else "",
                })
            df_meta = pd.DataFrame(meta_rows)
            df_pass = df_pass.merge(df_meta, on="code", how="left")
        else:
            df_pass["InstrumentName"] = ""
            df_pass["OpenDate"] = ""

        # ✅ 可选：上市满 N 天才纳入（只对拿得到 OpenDate 的生效）
        if self.min_listed_days is not None and self.min_listed_days > 0:
            od_ts = df_pass["OpenDate"].apply(lambda x: pd.to_datetime(x) if str(x).strip() else pd.NaT)
            listed_days = (self.end - od_ts).dt.days
            # OpenDate 为空的不强制剔除（你也可以改成强制剔除）
            mask_listed = od_ts.isna() | (listed_days >= self.min_listed_days)
            df_fail2 = df_pass[~mask_listed].copy()
            df_fail2["fail_reason"] = f"LISTED_LT_{self.min_listed_days}D"
            df_fail = pd.concat([df_fail, df_fail2], ignore_index=True)
            df_pass = df_pass[mask_listed].copy()

        # 排序
        df_pass = df_pass.sort_values(["coverage", "first_valid_date"], ascending=[False, True]).reset_index(drop=True)
        df_fail = df_fail.sort_values(["coverage", "first_valid_date"], ascending=[True, False]).reset_index(drop=True)

        codes = df_pass["code"].astype(str).tolist()
        return codes, df_pass, df_fail

    def save_csv(self, df_pass: pd.DataFrame, df_fail: pd.DataFrame) -> tuple[Path | None, Path | None]:
        if not self.out_dir:
            return None, None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        s = self.start.strftime("%Y%m%d")
        e = self.end.strftime("%Y%m%d")

        pass_path = self.out_dir / f"{self.out_tag}_pass_{s}_{e}.csv"
        fail_path = self.out_dir / f"{self.out_tag}_fail_{s}_{e}.csv"

        # Windows/Excel 友好
        df_pass.to_csv(pass_path, index=False, encoding="utf-8-sig")
        df_fail.to_csv(fail_path, index=False, encoding="utf-8-sig")

        return pass_path, fail_path
