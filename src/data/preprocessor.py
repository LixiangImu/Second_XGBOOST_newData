import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self):
        # 定义时间列映射
        self.time_columns = [
            ('排队日期', '排队时间'),
            ('叫号日期', '叫号时间'),
            ('入口操作日期', '入口操作时间'),
            ('回皮操作日期', '回皮操作时间'),
            ('出库操作日期', '出库操作时间'),
            ('出口操作日期', '出口操作时间')
        ]
        
        # 定义新列名映射
        self.datetime_columns = {
            ('排队日期', '排队时间'): '排队时刻',
            ('叫号日期', '叫号时间'): '叫号时刻',
            ('入口操作日期', '入口操作时间'): '入口操作时刻',
            ('回皮操作日期', '回皮操作时间'): '回皮操作时刻',
            ('出库操作日期', '出库操作时间'): '出库操作时刻',
            ('出口操作日期', '出口操作时间'): '出口操作时刻'
        }

    def load_data(self, file_path):
        """加载原始数据并进行基础检查"""
        df = pd.read_csv(file_path)
        print(f"原始数据记录数: {len(df)}")
        
        # 检查每个时间列的格式
        for date_col, time_col in self.time_columns:
            print(f"\n{date_col} - {time_col} 格式分析:")
            print(f"示例值: {df[date_col].iloc[0]} {df[time_col].iloc[0]}")
            print(f"时间格式样本:")
            print(df[time_col].unique()[:5])
        
        return df
    
    def combine_datetime(self, df):
        """合并日期时间列并转换为datetime格式"""
        try:
            for (date_col, time_col), new_col in self.datetime_columns.items():
                # 检查时间格式是否包含秒
                if ':' in df[time_col].iloc[0]:
                    if len(df[time_col].iloc[0].split(':')) == 3:
                        # 包含秒的格式
                        format_str = '%Y/%m/%d %H:%M:%S'
                    else:
                        # 只有时分的格式
                        format_str = '%Y/%m/%d %H:%M'
                else:
                    format_str = '%Y/%m/%d %H:%M'
                    
                print(f"Converting {date_col} + {time_col} using format: {format_str}")
                print(f"Sample datetime: {df[date_col].iloc[0]} {df[time_col].iloc[0]}")
                
                df[new_col] = pd.to_datetime(
                    df[date_col] + ' ' + df[time_col],
                    format=format_str
                )
            return df
        except Exception as e:
            print(f"日期转换错误，请检查数据格式: {str(e)}")
            print(f"示例日期时间: {df[date_col].iloc[0]} {df[time_col].iloc[0]}")
            raise

    def calculate_intervals(self, df):
        """计算排队到叫号的等待时间(分钟)"""
        # 只保留核心等待时间
        df['排队到叫号等待'] = (df['叫号时刻'] - df['排队时刻']).dt.total_seconds() / 60
        return df

    def add_time_features(self, df):
        """添加时间相关特征"""
        # 排队时刻的时间特征
        df['排队小时'] = df['排队时刻'].dt.hour
        df['排队分钟'] = df['排队时刻'].dt.minute
        df['排队星期'] = df['排队时刻'].dt.dayofweek
        df['是否周末'] = df['排队星期'].isin([5, 6]).astype(int)

        # 时间段划分
        df['时间段'] = pd.cut(
            df['排队小时'],
            bins=[-1, 6, 12, 18, 24],
            labels=['凌晨', '上午', '下午', '晚上']
        )

        return df
    
    def clean_data(self, df):
        """
        数据清洗
        """
        # 只保留排队到叫号等待时间的清洗
        df = df[df['排队到叫号等待'] >= 0]
        df = df[df['排队到叫号等待'] < 24*60]  # 排除等待超过24小时的记录
        
        # 处理缺失值
        df = df.dropna(subset=['排队到叫号等待'])
        
        return df
    
    def process(self, df):
        """
        完整的数据预处理流程
        """
        print("开始数据预处理...")
        
        df = self.combine_datetime(df)
        print("完成日期时间合并")
        
        df = self.calculate_intervals(df)
        print("完成时间间隔计算")
        
        df = self.add_time_features(df)
        print("完成时间特征提取")

        df = self.add_features(df)
        print("完成额外特征添加")
        
        df = self.clean_data(df)
        print(f"数据清洗后剩余记录数: {len(df)}")
        
        return df

    def add_features(self, df):
        """使用滑动窗口添加新特征"""
        # 确保排队时刻列存在且为datetime类型
        if '排队时刻' not in df.columns:
            print("警告：排队时刻列不存在，尝试重新创建...")
            df['排队时刻'] = pd.to_datetime(df['排队日期'] + ' ' + df['排队时间'])
        
        # 确保数据按时间排序并创建唯一索引
        df = df.sort_values('排队时刻').reset_index(drop=True)
        
        # 基础时间特征
        df['小时分钟'] = df['排队小时'] * 60 + df['排队分钟']
        df['是否高峰期'] = ((df['排队小时'] >= 8) & (df['排队小时'] <= 11)) | \
                          ((df['排队小时'] >= 14) & (df['排队小时'] <= 17))
        
        # 定义滑动窗口大小（分钟）
        windows = {
            '30分钟': 30,
            '1小时': 60,
            '2小时': 120,
            '4小时': 240
        }
        
        # 对每个煤种分别计算滑动窗口统计
        for window_name, minutes in windows.items():
            # 为每个煤种单独计算
            for coal_type in df['煤种编号'].unique():
                mask = df['煤种编号'] == coal_type
                coal_df = df[mask].copy()
                
                # 计算滑动窗口统计
                for idx in coal_df.index:
                    current_time = coal_df.loc[idx, '排队时刻']
                    window_start = current_time - pd.Timedelta(minutes=minutes)
                    
                    # 获取窗口内的数据
                    window_data = coal_df[
                        (coal_df['排队时刻'] > window_start) & 
                        (coal_df['排队时刻'] <= current_time)
                    ]
                    
                    # 只计算排队到叫号等待相关的统计量
                    df.loc[idx, f'{window_name}平均排队等待'] = window_data['排队到叫号等待'].mean()
                    df.loc[idx, f'{window_name}最大排队等待'] = window_data['排队到叫号等待'].max()
                    df.loc[idx, f'{window_name}标准差'] = window_data['排队到叫号等待'].std()
                    df.loc[idx, f'{window_name}煤种排队数'] = len(window_data)
            
            # 计算总体排队情况（不分煤种）
            for idx in df.index:
                current_time = df.loc[idx, '排队时刻']
                window_start = current_time - pd.Timedelta(minutes=minutes)
                
                # 计算窗口内的总排队数
                total_queue = len(df[
                    (df['排队时刻'] > window_start) & 
                    (df['排队时刻'] <= current_time)
                ])
                
                df.loc[idx, f'{window_name}总排队数'] = total_queue
        
        # 计算排队压力指数
        for window_name in windows.keys():
            df[f'{window_name}排队压力指数'] = (
                df[f'{window_name}煤种排队数'] / 
                df[f'{window_name}总排队数'].clip(lower=1)
            ) * df[f'{window_name}平均排队等待']
        
        # 添加时间周期特征
        df['��时周期性'] = np.sin(2 * np.pi * df['排队小时'] / 24)
        df['星期周期性'] = np.sin(2 * np.pi * df['排队星期'] / 7)
        
        # 填充可能的空值
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # 处理无穷大和NaN值
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
