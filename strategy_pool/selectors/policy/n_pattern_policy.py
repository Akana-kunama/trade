# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path

# ================= 路径 Hack (关键修复) =================
# 让脚本能找到项目根目录 D:\work\trade
# 当前文件在 strategy_pool/selectors/policy/ (3层深)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ================= 正常 Import =================
# 注意：是从 config.path_config 导入，而不是直接从 config 导入
from config.path_config import BASIC_INFO_DIR, FACTOR_STORE_DIR, MARKET_FACTOR_PATH
from strategy_pool.selectors.policy.base_selector import SelectorBase

# 引入 AI 工厂组件
from factor_lab.service.data_provider import FeatureService
from model_factory.registry import ModelRegistry

class NPatternSelector(SelectorBase):
    def __init__(self):
        super().__init__(strategy_name="n_pattern_rebound")
        
        # 1. 基础服务初始化
        self.svc = FeatureService(str(FACTOR_STORE_DIR), str(MARKET_FACTOR_PATH))
        self.registry = ModelRegistry()
        
        # 2. 加载基础信息 (用于ST过滤)
        self.stock_info_path = BASIC_INFO_DIR / "stock.csv"
        self.valid_whitelist, self.info_df = self._load_stock_info()

        # 3. [自动加载] 加载 n_pattern 系列最新的三个模型
        self.models = {}
        self.features = [] 
        self._load_latest_models()

    def _load_stock_info(self):
        """读取 ST 状态 / 白名单"""
        if not self.stock_info_path.exists():
            print(f"⚠️ [Warn] 股票信息表不存在: {self.stock_info_path}")
            return set(), pd.DataFrame()
        try:
            # 兼容读取
            try:
                df = pd.read_csv(self.stock_info_path, dtype={'code': str}, encoding='utf-8')
            except:
                df = pd.read_csv(self.stock_info_path, dtype={'code': str}, encoding='gbk')
                
            # 假设 CSV 里有 special_type 列，没有就全都要
            if 'special_type' in df.columns:
                df = df[df['special_type'] == 'Normal']
            
            # 标准化 code 为 6位字符串
            if 'code' in df.columns:
                valid_codes = set(df['code'].apply(lambda x: str(x)[:6]))
                return valid_codes, df
            return set(), df
        except Exception as e:
            print(f"⚠️ 加载股票信息出错: {e}")
            return set(), pd.DataFrame()

    def _load_latest_models(self):
        """自动去 Model Zoo 找最新的三个模型"""
        target_models = {
            "lgbm": "n_pattern/v1_lgbm",
            "xgb":  "n_pattern/v1_xgb",
            "cat":  "n_pattern/v1_cat"
        }
        
        first_load = True
        for name, prefix in target_models.items():
            # 自动查找最新ID
            latest_id = self.registry.get_latest_id(prefix)
            
            if not latest_id:
                print(f"❌ [Error] 找不到 {name} 的模型！请先运行 scripts/train_n_pattern.py")
                continue
                
            try:
                # 加载
                model, meta = self.registry.load_model(latest_id)
                self.models[name] = model
                
                if first_load:
                    self.features = meta['config']['features']
                    first_load = False
                print(f"✅ 模型 {name} 加载成功")
            except Exception as e:
                print(f"❌ 模型 {name} 加载失败: {e}")

    def run(self, date=None):
        """
        选股主入口
        """
        # 如果没传日期，默认今天
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\n>>> [AI Strategy] 启动 N字反转选股: {date}")
        
        # 1. 扫描所有股票代码
        all_codes = [f.stem for f in FACTOR_STORE_DIR.glob("*.parquet")]
        print(f"📋 扫描因子库，共 {len(all_codes)} 只股票")
        target_codes = [c for c in all_codes if re.match(r'^(00|60)', c)]
        print(f"📋 扫描主板股票: {len(target_codes)} 只 (已过滤300/688)")
        # 2. 获取数据 (Features + Signal)
        if not self.features:
            print("❌ 模型未加载成功，无法获取特征列表")
            return

        req_cols = list(set(self.features + ['n_pattern_signal', 'close']))
        
        try:
            df_today = self.svc.get_features(
                codes=target_codes,
                factor_names=req_cols,
                start_date=date,
                end_date=date
            )
        except Exception as e:
            print(f"❌ 获取特征数据失败: {e}")
            return
        
        if df_today.empty:
            print(f"⚠️ 日期 {date} 无数据 (可能是非交易日或数据未更新)")
            return

        # 3. [硬过滤] N字反转信号
        if 'n_pattern_signal' in df_today.columns:
            # 筛选 signal == 1
            df_candidates = df_today[df_today['n_pattern_signal'] == 1].copy()
        else:
            print("❌ 因子库中缺少 n_pattern_signal 列")
            return

        print(f"🔍 符合N字形态初选: {len(df_candidates)} 只")
        if df_candidates.empty:
            return

        # 4. [过滤] ST 黑名单
        if self.valid_whitelist:
            # 提取 6 位代码
            df_candidates['code_6'] = df_candidates['code'].apply(lambda x: x[:6])
            df_candidates = df_candidates[df_candidates['code_6'].isin(self.valid_whitelist)]
            print(f"🛡️ 去除ST后剩余: {len(df_candidates)} 只")

        if df_candidates.empty:
            return

        # 5. [AI预测] 三模型打分
        X = df_candidates[self.features].fillna(0)
        
        top_k_per_model = 10
        selected_indices = set()
        
        # 记录每个模型的打分
        for name, model in self.models.items():
            scores = model.predict(X)
            col_name = f'score_{name}'
            df_candidates[col_name] = scores
            
            # 取该模型的 Top K
            top_df = df_candidates.nlargest(top_k_per_model, col_name)
            # selected_indices.update(top_df.index.tolist())
            selected_indices.update(top_df['code'].tolist())

            print(f"   🤖 {name} 推荐: {top_df['code'].tolist()}")

        # 6. [并集] 取三个模型 Top K 的并集作为最终池
        # final_pool = df_candidates.loc[list(selected_indices)].copy()
        final_pool = df_candidates[df_candidates['code'].isin(selected_indices)].copy()

        # 计算平均分用于最终排序
        score_cols = [c for c in final_pool.columns if c.startswith('score_')]
        final_pool['score_avg'] = final_pool[score_cols].mean(axis=1)
        
        # 排序
        final_pool.sort_values(by='score_avg', ascending=False, inplace=True)
        
        # 7. 格式化输出
        result_df = final_pool[['code', 'score_avg', 'close']].copy()
        result_df['reason'] = 'AI_N_Pattern_Ensemble'
        result_df['date'] = date
        
        print(f"\n🎉 最终入选股票池 ({len(result_df)} 只):")
        print(result_df)
        
        # 8. 保存结果
        # 8.1. 构造文件名
        if isinstance(date, str):
            date_str = date.replace('-', '')
        else:
            date_str = date.strftime('%Y%m%d')
            
        filename = f"{date_str}.csv"
        
        # 8.2. 构造保存路径 (使用 config 里的 STRATEGY_WORKSPACE)
        from config.path_config import STRATEGY_WORKSPACE
        # 策略子文件夹
        save_dir = STRATEGY_WORKSPACE / self.strategy_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / filename
        
        # 8.3. 保存
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"💾 结果已保存至: {save_path}")
        # 9. 保存 (父类方法)
        self.save_result(result_df)

if __name__ == "__main__":
    # 这里的 hack 是为了让单独右键运行该文件也能成功
    # 实际上应该通过 run_daily_selection.py 运行
    s = NPatternSelector()
    # 找一个最近的有数据的日期测试 (例如上周五)
    s.run(date="2025-12-18") 
    # s.run() # 默认跑今天