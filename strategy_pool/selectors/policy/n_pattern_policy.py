# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path

# ================= 路径 Hack =================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ================= Imports =================
from config.path_config import BASIC_INFO_DIR, FACTOR_STORE_DIR, MARKET_FACTOR_PATH
from strategy_pool.selectors.policy.base_selector import SelectorBase

from factor_lab.service.data_provider import FeatureService
from model_factory.registry import ModelRegistry
from toolbox.messengers.wechat_bot import WechatBot

class NPatternSelector(SelectorBase):
    def __init__(self):
        super().__init__(strategy_name="n_pattern_rebound")
        
        self.svc = FeatureService(str(FACTOR_STORE_DIR), str(MARKET_FACTOR_PATH))
        self.registry = ModelRegistry()
        self.bot = WechatBot()
        
        self.stock_info_path = BASIC_INFO_DIR / "stock.csv"
        self.valid_whitelist, self.info_df = self._load_stock_info()

        self.models = {}
        self.features = [] 
        self._load_latest_models()

    def _load_stock_info(self):
        """加载基础信息"""
        if not self.stock_info_path.exists():
            return set(), pd.DataFrame()
        try:
            try:
                df = pd.read_csv(self.stock_info_path, dtype={'order_book_id': str}, encoding='utf-8')
            except:
                df = pd.read_csv(self.stock_info_path, dtype={'order_book_id': str}, encoding='gbk')
            
            if 'order_book_id' in df.columns:
                df['code_key'] = df['order_book_id'].astype(str).str[:6]
            else:
                return set(), pd.DataFrame()

            if 'special_type' in df.columns:
                normal_df = df[df['special_type'] == 'Normal'].copy()
            else:
                normal_df = df.copy()
            
            cols_needed = ['code_key', 'symbol', 'industry_name']
            cols_exist = [c for c in cols_needed if c in normal_df.columns]
            info_df = normal_df[cols_exist].copy()
            
            valid_codes = set(normal_df['code_key'].values)
            return valid_codes, info_df

        except Exception:
            return set(), pd.DataFrame()

    def _load_latest_models(self):
        """加载6个模型"""
        target_prefixes = {
            "label1_lgbm": "n_pattern/v1_lgbm_label_strategy",
            "label1_xgb":  "n_pattern/v1_xgb_label_strategy",
            "label1_cat":  "n_pattern/v1_cat_label_strategy",
            "label2_lgbm": "n_pattern/v1_lgbm_label_short_term",
            "label2_xgb":  "n_pattern/v1_xgb_label_short_term",
            "label2_cat":  "n_pattern/v1_cat_label_short_term",
        }
        
        first_load = True
        print("📥 正在加载 AI 模型组...")
        for name, prefix in target_prefixes.items():
            latest_id = self.registry.get_latest_id(prefix)
            if not latest_id:
                print(f"   ⚠️ 缺失: {name}")
                continue
            try:
                model, meta = self.registry.load_model(latest_id)
                self.models[name] = model
                if first_load:
                    self.features = meta['config']['features']
                    first_load = False
            except Exception as e:
                print(f"   ❌ 加载失败 {name}: {e}")

    def run(self, date=None):
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n>>> [AI Strategy] 启动 N字反转双模选股: {date}")
        

        # 1. 扫描与过滤代码
        all_codes = [f.stem for f in FACTOR_STORE_DIR.glob("*.parquet")]
        target_codes = [c for c in all_codes if re.match(r'^(00|60)', c)]
        
        # 2. 取数
        req_cols = list(set(self.features + ['n_pattern_signal', 'close']))
        try:
            df_today = self.svc.get_features(
                codes=target_codes,
                factor_names=req_cols,
                start_date=date,
                end_date=date
            )
        except Exception:
            return

        if df_today.empty:
            print(f"⚠️ {date} 无数据")
            return

        # 3. 筛选 N字反转
        if 'n_pattern_signal' not in df_today.columns:
            print("❌ 缺失 n_pattern_signal")
            return
            
        df_candidates = df_today[df_today['n_pattern_signal'] == 1].copy()
        print(f"🔍 N字反转初选: {len(df_candidates)} 只")
        
        if df_candidates.empty:
            return

        # 4. 筛选 ST
        if self.valid_whitelist:
            df_candidates['code_6'] = df_candidates['code'].apply(lambda x: x[:6])
            df_candidates = df_candidates[df_candidates['code_6'].isin(self.valid_whitelist)]
            print(f"🛡️ 去除ST后剩余: {len(df_candidates)} 只")

        if df_candidates.empty:
            return

        # 5. AI 打分
        X = df_candidates[self.features].fillna(0)
        for name, model in self.models.items():
            df_candidates[f'score_{name}'] = model.predict(X)

        # 6. [核心修正] 集合运算 (基于 Code 字符串，而非 Index)
        TOP_K_PER_MODEL = 10
        TARGET_POOL_SIZE = 13 # 目标总数：共振 + 补录
        
        # --- 集合 1: Label Strategy 的代码集 ---
        set1_codes = set()
        for algo in ['lgbm', 'xgb', 'cat']:
            key = f'score_label1_{algo}'
            if key in df_candidates.columns:
                top = df_candidates.nlargest(TOP_K_PER_MODEL, key)
                set1_codes.update(top['code'].tolist()) # [Fix] 用 code
        
        # --- 集合 2: Label ShortTerm 的代码集 ---
        set2_codes = set()
        for algo in ['lgbm', 'xgb', 'cat']:
            key = f'score_label2_{algo}'
            if key in df_candidates.columns:
                top = df_candidates.nlargest(TOP_K_PER_MODEL, key)
                set2_codes.update(top['code'].tolist()) # [Fix] 用 code
        
        # --- 集合 3: 交集 (共振) ---
        resonance_codes = list(set1_codes & set2_codes)
        
        # 计算加权平均分 (所有6个模型)
        score_cols = [c for c in df_candidates.columns if c.startswith('score_')]
        df_candidates['final_score'] = df_candidates[score_cols].mean(axis=1)
        
        final_selected_codes = [] # 最终所有入选的代码 (用于生成CSV)
        
        # === 阶段 A: 共振股 (Message 1) ===
        df_resonance = pd.DataFrame()
        msg_resonance = ""
        
        if resonance_codes:
            # 使用 code 列进行筛选 [Fix]
            df_resonance = df_candidates[df_candidates['code'].isin(resonance_codes)].copy()
            df_resonance.sort_values(by='final_score', ascending=False, inplace=True)
            
            final_selected_codes.extend(df_resonance['code'].tolist())
            msg_resonance = self._format_wechat_msg("🚀【双模共振股】(置信度高)", df_resonance)
            print(f"🎯 共振股: {len(df_resonance)} 只")
        else:
            msg_resonance = "🚀【双模共振股】\n今日无双模共振标的"

        # === 阶段 B: 补录股 (Message 2) ===
        remaining_slots = TARGET_POOL_SIZE - len(final_selected_codes)
        df_supplement = pd.DataFrame()
        msg_supplement = ""
        
        if remaining_slots > 0:
            # 排除已选的 [Fix: 使用 ~isin]
            candidates_left = df_candidates[~df_candidates['code'].isin(final_selected_codes)]
            
            # 策略：优先从两个集合的并集里找，按分高低补录
            union_codes = list(set1_codes | set2_codes)
            # 在剩余池中，属于并集的
            df_union_left = candidates_left[candidates_left['code'].isin(union_codes)]
            
            # 先从并集剩余里选
            supplement_1 = df_union_left.nlargest(remaining_slots, 'final_score')
            
            # 如果还不够，从全量剩余里选
            needed_more = remaining_slots - len(supplement_1)
            supplement_2 = pd.DataFrame()
            if needed_more > 0:
                others = candidates_left.drop(supplement_1.index) # 这里可以用index drop因为是同一df
                supplement_2 = others.nlargest(needed_more, 'final_score')
                
            df_supplement = pd.concat([supplement_1, supplement_2])
            final_selected_codes.extend(df_supplement['code'].tolist())
            
            msg_supplement = self._format_wechat_msg("👀【补充关注股】(模型高分)", df_supplement)
            print(f"📉 补录股: {len(df_supplement)} 只")
        else:
            msg_supplement = "👀【补充关注股】\n共振股已满额，无需补录"

        # 7. 生成最终 CSV (包含共振+补录，共10只左右)
        # [Fix] 使用 isin(final_selected_codes)
        final_df = df_candidates[df_candidates['code'].isin(final_selected_codes)].copy()
        final_df.sort_values(by='final_score', ascending=False, inplace=True)
        
        # 关联信息
        final_df['code_key'] = final_df['code'].apply(lambda x: x[:6])
        if not self.info_df.empty:
            final_df = pd.merge(final_df, self.info_df, on='code_key', how='left')
            final_df['symbol'] = final_df['symbol'].fillna(final_df['code'])
            final_df['industry_name'] = final_df['industry_name'].fillna('-')
        else:
            final_df['symbol'] = final_df['code']
            final_df['industry_name'] = '-'

        # 整理输出
        output_cols = ['code', 'symbol', 'final_score', 'industry_name', 'close']
        result_df = final_df[output_cols].copy()
        result_df.rename(columns={'final_score': 'score'}, inplace=True)
        
        print("\n🎉 最终输出池:")
        print(result_df)

        # 8. 保存与发送
        self._save_to_csv(result_df, date)
        
        print("\n📨 推送微信...")
        self.bot.send_text(msg_resonance)
        self.bot.send_text(msg_supplement)

    def _format_wechat_msg(self, title, df):
        if df.empty: return f"{title}\n无"
        
        # 临时 merge symbol
        df_temp = df.copy()
        df_temp['code_key'] = df_temp['code'].apply(lambda x: x[:6])
        if not self.info_df.empty:
            df_temp = pd.merge(df_temp, self.info_df, on='code_key', how='left')
            df_temp['symbol'] = df_temp['symbol'].fillna('')
            df_temp['industry_name'] = df_temp['industry_name'].fillna('')
        else:
            df_temp['symbol'] = df_temp['code']
            df_temp['industry_name'] = '-'
            
        msg = [title]
        # 优化排版：代码 | 名称 | 分数
        msg.append(f"{'代码':<7} {'名称':<5} {'分':<4}")
        msg.append("-" * 20)
        
        for _, row in df_temp.iterrows():
            c = row['code'][:6]
            n = row['symbol'][:4]
            s = f"{row['final_score']:.2f}"
            msg.append(f"{c} {n} {s}")
            
        return "\n".join(msg)

    def _save_to_csv(self, df, date_str):
        from config.path_config import STRATEGY_WORKSPACE
        clean_date = date_str.replace('-', '')
        save_dir = STRATEGY_WORKSPACE / self.strategy_name
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"{clean_date}.csv"
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"💾 保存: {file_path}")


if __name__ == "__main__":
    s = NPatternSelector()
    s.run(date="2025-12-18")