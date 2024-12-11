# src/main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib
import logging

from config import Config
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        Config.DATA_DIR / 'raw',
        Config.DATA_DIR / 'processed',
        Config.DATA_DIR / 'features',
        Config.MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def prepare_data():
    """数据准备流程"""
    logger.info("开始数据准备流程...")
    
    # 1. 数据预处理
    preprocessor = DataPreprocessor()
    raw_data = preprocessor.load_data(Config.RAW_DATA_PATH)
    processed_data = preprocessor.process(raw_data)
    
    # 保存处理后的数据
    processed_data.to_csv(Config.PROCESSED_DATA_PATH, index=False)
    logger.info(f"处理后的数据已保存至: {Config.PROCESSED_DATA_PATH}")
    
    # 2. 特征工程
    feature_engineer = FeatureEngineer()
    features = feature_engineer.process(processed_data)
    
    # 准备目标变量
    targets = processed_data[['排队到叫号等待']]
    
    # 保存特征和标签
    features.to_csv(Config.FEATURES_DATA_PATH, index=False)
    targets.to_csv(Config.DATA_DIR / 'features' / 'targets.csv', index=False)
    
    # 保存特征工程器（用于后续预测）
    joblib.dump(feature_engineer, Config.MODELS_DIR / 'feature_engineer.joblib')
    
    return features, targets

def data_analysis(processed_data):
    """数据分析"""
    logger.info("\n=== 数据分析报告 ===")
    
    # 基本统计信息
    logger.info("\n等待时间统计(分钟):")
    wait_time_stats = processed_data[['排队到叫号等待']].describe()
    logger.info(f"\n{wait_time_stats}")
    
    # 煤种分布
    logger.info("\n煤种分布:")
    coal_type_dist = processed_data['煤种编号'].value_counts()
    logger.info(f"\n{coal_type_dist}")
    
    # 时间段分布
    logger.info("\n时间段分布:")
    time_period_dist = processed_data['时间段'].value_counts()
    logger.info(f"\n{time_period_dist}")
    
    # 每小时平均等待时间
    hourly_wait = processed_data.groupby('排队小时')[['排队到叫号等待']].mean()
    logger.info("\n每小时平均等待时间:")
    logger.info(f"\n{hourly_wait}")

def split_data(features, targets):
    """划分训练集和测试集"""
    # 首先划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        targets,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED
    )
    
    # 从训练集中划分出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=Config.VALIDATION_SIZE,
        random_state=Config.RANDOM_SEED
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """主函数"""
    try:
        # 1. 创建目录
        setup_directories()
        logger.info("目录结构创建完成")
        
        # 2. 数据准备
        features, targets = prepare_data()
        logger.info("数据准备完成")
        
        # 3. 数据分析
        processed_data = pd.read_csv(Config.PROCESSED_DATA_PATH)
        data_analysis(processed_data)
        logger.info("数据分析完成")
        
        # 4. 数据集划分
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, targets)
        logger.info(f"训练集大小: {len(X_train)}")
        logger.info(f"验证集大小: {len(X_val)}")
        logger.info(f"测试集大小: {len(X_test)}")
        
        # 保存数据集划分
        train_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        for name, data in train_data.items():
            data.to_csv(Config.DATA_DIR / 'processed' / f'{name}.csv', index=False)
        
        logger.info("数据集划分完成并保存")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()