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
        """计算各个环节的时间间隔(分钟)"""
        # 核心等待时间
        df['排队到叫号等待'] = (df['叫号时刻'] - df['排队时刻']).dt.total_seconds() / 60
        df['叫号到入口等待'] = (df['入口操作时刻'] - df['叫号时刻']).dt.total_seconds() / 60  # 修正列名

        # 其他流程时间
        df['入口到回皮'] = (df['回皮操作时刻'] - df['入口操作时刻']).dt.total_seconds() / 60  # 修正列名
        df['回皮到出库'] = (df['出库操作时刻'] - df['回皮操作时刻']).dt.total_seconds() / 60  # 修正列名
        df['出库到出口'] = (df['出口操作时刻'] - df['出库操作时刻']).dt.total_seconds() / 60  # 修正列名
        df['总处理时间'] = (df['出口操作时刻'] - df['入口操作时刻']).dt.total_seconds() / 60  # 修正列名
    
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
        # 移除明显异常值
        df = df[df['排队到叫号等待'] >= 0]
        df = df[df['排队到叫号等待'] < 24*60]  # 排除等待超过24小时的记录
        df = df[df['叫号到入口等待'] >= 0]
        df = df[df['叫号到入口等待'] < 12*60]  # 排除等待超过12小时的记录
        
        # 处理缺失值
        df = df.dropna(subset=['排队到叫号等待', '叫号到入口等待'])
        
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
        
        df = self.clean_data(df)
        print(f"数据清洗后剩余记录数: {len(df)}")
        
        return df

    def add_features(self, df):
        """添加新特征"""
        # 时间相关特征
        df['小时分钟'] = df['排队小时'] * 60 + df['排队分钟']  # 转换为分钟计数
        df['是否高峰期'] = ((df['排队小时'] >= 8) & (df['排队小时'] <= 11)) | \
                          ((df['排队小时'] >= 14) & (df['排队小时'] <= 17))
        
        # 排队相关特征
        df['单位煤种平均排队数'] = df['同煤种排队数'] / df.groupby('煤种编号_encoded')['同煤种排队数'].transform('count')
        df['排队压力指数'] = df['当前排队数'] * df['同煤种排队数'] / df['总排队数'].mean()
        
        # 历史等待时间的统计特征
        for col in ['前1小时平均排队等待', '前2小时平均排队等待', '前4小时平均排队等待']:
            df[f'{col}_std'] = df.groupby('煤种编号_encoded')[col].transform('std')
            df[f'{col}_max'] = df.groupby('煤种编号_encoded')[col].transform('max')
        
        return df
