# src/config.py
from pathlib import Path

class Config:
    # 路径配置
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    RAW_DATA_PATH = DATA_DIR / 'raw' / 'queue_data_2050.csv'
    PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'processed_data.csv'
    FEATURES_DATA_PATH = DATA_DIR / 'features' / 'features.csv'
    MODELS_DIR = ROOT_DIR / 'models' 
    
    # 数据处理配置
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    # 模型配置
    MODEL_PARAMS = {
        'learning_rate': 0.01,
        'max_depth': 6,
        'n_estimators': 100,
        'objective': 'reg:squarederror'
    }
    
    # 训练配置
    EPOCHS = 100
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 10
    
    # 预测配置
    UPDATE_INTERVAL = 5  # 分钟