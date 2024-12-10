import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from config import Config

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.feature_columns = []
        
    def create_queue_features(self, df):
        """
        创建排队相关特征
        """
        # 按时间排序
        df = df.sort_values('排队时刻')
        
        # 计算当前队列长度
        df['当前排队数'] = 0
        for idx in df.index:
            current_time = df.loc[idx, '排队时刻']
            mask = (df['排队时刻'] <= current_time) & (df['叫号时刻'] > current_time)
            df.loc[idx, '当前排队数'] = mask.sum()
            
        # 计算各煤种的等待车辆数
        df['同煤种排队数'] = 0
        for idx in df.index:
            current_time = df.loc[idx, '排队时刻']
            current_coal = df.loc[idx, '煤种编号']
            mask = (
                (df['排队时刻'] <= current_time) & 
                (df['叫号时刻'] > current_time) & 
                (df['煤种编号'] == current_coal)
            )
            df.loc[idx, '同煤种排队数'] = mask.sum()
            
        return df
    
    def create_historical_features(self, df):
        """
        创建历史统计特征
        """
        # 计算最近N小时的平均等待时间
        for hours in [1, 2, 4]:
            # 排队到叫号
            df[f'前{hours}小时平均排队等待'] = df.rolling(
                window=f'{hours}H',
                on='排队时刻'
            )['排队到叫号等待'].mean()
            
            # 叫号到入口
            df[f'前{hours}小时平均入口等待'] = df.rolling(
                window=f'{hours}H',
                on='排队时刻'
            )['叫号到入口等待'].mean()
        
        # 填充缺失值
        historical_cols = [col for col in df.columns if '前' in col and '小时' in col]
        df[historical_cols] = df[historical_cols].fillna(method='bfill')
        
        return df
    
    def encode_categorical_features(self, df):
        """
        对分类特征进行编码
        """
        categorical_columns = ['煤种编号', '时间段']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def select_features(self, df):
        """
        选择最终使用的特征
        """
        self.feature_columns = [
            '排队小时', '排队分钟', '排队星期', '是否周末',
            '当前排队数', '同煤种排队数',
            '前1小时平均排队等待', '前2小时平均排队等待', '前4小时平均排队等待',
            '前1小时平均入口等待', '前2小时平均入口等待', '前4小时平均入口等待',
            '煤种编号_encoded', '时间段_encoded'
        ]
        
        return df[self.feature_columns]
    
    def process(self, df):
        """
        完整的特征工程流程
        """
        print("开始特征工程...")
        
        df = self.create_queue_features(df)
        print("完成队列特征创建")
        
        df = self.create_historical_features(df)
        print("完成历史特征创建")
        
        df = self.encode_categorical_features(df)
        print("完成分类特征编码")
        
        features = self.select_features(df)
        print(f"最终特征数量: {len(self.feature_columns)}")
        
        return features
    
    def process_single(self, data):
        """处理单条预测数据"""
        # 获取历史统计数据（这部分需要预先计算好并保存）
        historical_stats = joblib.load(Config.MODELS_DIR / 'historical_stats.joblib')
        
        # 合并历史统计特征
        features = data.merge(
            historical_stats,
            on=['煤种编号', '排队小时'],
            how='left'
        )
        
        # 填充缺失值
        features = features.fillna(historical_stats.mean())
        
        return features
