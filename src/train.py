# src/train.py
import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import Config
from models.queue_predictor import XGBoostQueuePredictor
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """加载预处理后的数据"""
    X_train = pd.read_csv(Config.DATA_DIR / 'processed' / 'X_train.csv')
    X_val = pd.read_csv(Config.DATA_DIR / 'processed' / 'X_val.csv')
    X_test = pd.read_csv(Config.DATA_DIR / 'processed' / 'X_test.csv')
    y_train = pd.read_csv(Config.DATA_DIR / 'processed' / 'y_train.csv')
    y_val = pd.read_csv(Config.DATA_DIR / 'processed' / 'y_val.csv')
    y_test = pd.read_csv(Config.DATA_DIR / 'processed' / 'y_test.csv')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_features(X_train, X_val, X_test):
    """更新特征预处理"""
    scaler = StandardScaler()
    
    # 获取实际存在的数值型特征
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 排除已经编码的分类特征（如果有的话）
    numeric_features = [col for col in numeric_features if not col.endswith('_encoded')]
    
    logger.info("对以下特征进行标准化处理:")
    logger.info(f"特征列表: {numeric_features}")
    
    # 对训练集进行拟合和转换
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    
    # 对验证集和测试集只进行转换
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    X_val_scaled[numeric_features] = scaler.transform(X_val[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    
    # 保存scaler
    joblib.dump(scaler, Config.MODELS_DIR / 'scaler.joblib')
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def analyze_data(X_train, y_train):
    """分析训练数据"""
    logger.info("\n数据分析报告:")
    
    # 特征统计
    logger.info("\n特征统计:")
    logger.info(f"数值型特征:\n{X_train.describe()}")
    
    # 目标变量统计
    logger.info("\n目标变量统计:")
    logger.info(f"\n{y_train.describe()}")
    
    # 相关性分析
    correlation = pd.concat([X_train, y_train], axis=1).corr()
    logger.info("\n目标变量与特征的相关性:")
    for target in y_train.columns:
        correlations = correlation[target].sort_values(ascending=False)
        logger.info(f"\n{target} - Top 5 相关特征:")
        logger.info(correlations.head())

def train_model():
    """训练模型的主函数"""
    logger.info("开始模型训练流程...")
    
    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    logger.info(f"数据加载完成. 训练集大小: {len(X_train)}")
    
    # 2. 数据分析
    analyze_data(X_train, y_train)
    
    # 3. 特征预处理
    X_train_scaled, X_val_scaled, X_test_scaled = preprocess_features(
        X_train, X_val, X_test
    )
    logger.info("特征预处理完成")
    
    # 4. 创建并训练模型
    model = XGBoostQueuePredictor()
    
    # 选择训练方式：标准训练或带早停的训练
    if X_val is not None and y_val is not None:
        logger.info("使用带早停的训练方式...")
        model.fit_with_early_stopping(X_train_scaled, y_train, X_val_scaled, y_val)
        train_metrics = model.evaluate(X_train_scaled, y_train)
    else:
        logger.info("使用标准训练方式...")
        train_metrics = model.train(X_train_scaled, y_train)
    
    # 5. 在测试集上评估
    test_metrics = model.evaluate(X_test_scaled, y_test)
    model.log_metrics(test_metrics, prefix='Test')
    
    # 保存评估报告
    report_dir = Config.MODELS_DIR / 'evaluation_reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / 'test_evaluation_report.txt', 'w') as f:
        f.write(f"Test Metrics:\n{test_metrics}\n")
    
    # 6. 记录特征重要性
    model.log_feature_importance()
    
    # 7. 保存结果
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': model.get_feature_importance()
    }
    
    # 创建保存目录（如果不存在）
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存结果和模型
    joblib.dump(results, Config.MODELS_DIR / 'training_results.joblib')
    
    # 保存模型到单独的文件夹
    model_dir = Config.MODELS_DIR / 'trained_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'queue_predictor.joblib')
    
    logger.info("训练完成！模型和结果已保存。")
    
    return model, results

def train_with_cv(X, y, n_splits=5):
    """使用交叉验证训练模型"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBoostQueuePredictor()
        model.fit_with_early_stopping(X_train, y_train, X_val, y_val)
        
        val_metrics = model.evaluate(X_val, y_val)
        cv_scores.append(val_metrics)
        
        # 保存每个fold的评估报告
        report_dir = Config.MODELS_DIR / 'evaluation_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / f'fold_{fold}_evaluation_report.txt', 'w') as f:
            f.write(f"Fold {fold} Validation Metrics:\n{val_metrics}\n")
        
        # 保存每个fold的模型
        model_dir = Config.MODELS_DIR / 'trained_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f'fold_{fold}_queue_predictor.joblib')
        
        logger.info(f"\nFold {fold} 验证集评估结果:")
        model.log_metrics(val_metrics, prefix=f'Fold {fold}')
    
    return cv_scores

if __name__ == "__main__":
    train_model()