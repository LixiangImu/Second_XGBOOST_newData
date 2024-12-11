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
    
    def create_cyclical_features(self, df):
        """创建周期性特征"""
        # 添加小时的周期性特征
        df['小时周期性'] = np.sin(2 * np.pi * df['排队小时'] / 24)
        # 添加星期的周期性特征
        df['星期周期性'] = np.sin(2 * np.pi * df['排队星期'] / 7)
        return df
    
    def process(self, df):
        """
        完整的特征工程流程
        """
        print("开始特征工程...")
        
        # 创建队列特征
        df = self.create_queue_features(df)
        print("完成队列特征创建")
        
        # 创建历史特征
        df = self.create_historical_features(df)
        print("完成历史特征创建")
        
        # 创建周期性特征
        df = self.create_cyclical_features(df)
        print("完成周期性特征创建")
        
        # 编码分类特征
        df = self.encode_categorical_features(df)
        print("完成分类特征编码")
        
        features = self.select_features(df)
        print(f"最终特征数量: {len(self.feature_columns)}")
        
        return features
    
    def select_features(self, df):
        """更新特征选择"""
        self.feature_columns = [
            # 基础时间特征
            '排队小时', '排队分钟', '排队星期', '是否周末',
            '小时分钟', '是否高峰期',
            '小时周期性', '星期周期性',  # 确保这些特征已经被创建
            
            # 队列特征
            '当前排队数', '同煤种排队数',
            
            # 历史特征
            '前1小时平均排队等待', '前2小时平均排队等待', '前4小时平均排队等待',
            
            # 30分钟窗口特征
            '30分钟平均排队等待', '30分钟最大排队等待', '30分钟标准差',
            '30分钟煤种排队数', '30分钟总排队数', '30分钟排队压力指数',
            
            # 1小时窗口特征
            '1小时平均排队等待', '1小时最大排队等待', '1小时标准差',
            '1小时煤种排队数', '1小时总排队数', '1小时排队压力指数',
            
            # 2小时窗口特征
            '2小时平均排队等待', '2小时最大排队等待', '2小时标准差',
            '2小时煤种排队数', '2小时总排队数', '2小时排队压力指数',
            
            # 4小时窗口特征
            '4小时平均排队等待', '4小时最大排队等待', '4小时标准差',
            '4小时煤种排队数', '4小时总排队数', '4小时排队压力指数',
            
            # 分类特征编码
            '煤种编号_encoded', '时间段_encoded'
        ]
        
        # 检查所有特征是否都存在
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise KeyError(f"以下特征在数据中不存在: {missing_features}")
        
        return df[self.feature_columns]
    
    def process_single(self, data):
        """修改单条预测处理逻辑"""
        # 获取历史数据
        historical_data = pd.read_csv(Config.PROCESSED_DATA_PATH)
        historical_data['排队时刻'] = pd.to_datetime(historical_data['排队时刻'])
        
        # 计算滑动窗口特征
        current_time = pd.to_datetime(data['排队时刻'].iloc[0])
        coal_type = data['煤种编号'].iloc[0]
        
        # 计算各个时间窗口的特征
        windows = {
            '30分钟': '30min',
            '1小时': '1H',
            '2小时': '2H',
            '4小时': '4H'
        }
        
        window_features = {}
        for window_name, window_size in windows.items():
            window_data = historical_data[
                (historical_data['排队时刻'] <= current_time) &
                (historical_data['排队时刻'] >= current_time - pd.Timedelta(window_size))
            ]
            
            window_features.update(self._calculate_window_stats(
                window_data, window_name, coal_type
            ))
        
        # 合并所有特征
        data = data.assign(**window_features)
        
        # 编码分类特征
        data = self.encode_categorical_features(data)
        
        return self.select_features(data)
