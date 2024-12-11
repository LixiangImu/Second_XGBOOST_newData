# src/train.py
import logging
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from config import Config
from models.queue_predictor import XGBoostQueuePredictor
from sklearn.model_selection import KFold
from datetime import datetime
import numpy as np

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

def plot_learning_curves(model, X_train, y_train, X_val, y_val, run_dir):
    """绘制学习曲线"""
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建评估报告目录
    report_dir = run_dir / 'evaluation_reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成学习曲线数据
    train_sizes, train_scores, val_scores = learning_curve(
        model.base_model,
        X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    # 计算均值和标准差
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = -np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Set')
    plt.plot(train_sizes, val_mean, label='Validation Set')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 保存图片到评估报告目录
    plt.savefig(report_dir / 'learning_curves.png')
    plt.close()
    
    logger.info(f"\n学习曲线已保存至: {report_dir / 'learning_curves.png'}")

def check_overfitting(model, X_train, y_train, X_val, y_val, X_test, y_test, run_dir):
    """检查模型过拟合情况"""
    # 创建评估报告目录
    report_dir = run_dir / 'evaluation_reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n=== 过拟合分析 ===")
    
    # 在所有数据集上评估
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)
    
    # 准备评估报告内容
    report_content = ["=== 过拟合分析报告 ===\n"]
    
    # 打印比较结果
    report_content.append("\n性能比较:")
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    datasets = ['训练集', '验证集', '测试集']
    results = [train_metrics, val_metrics, test_metrics]
    
    for metric in metrics:
        report_content.append(f"\n{metric}:")
        for dataset, result in zip(datasets, results):
            value = result['排队到叫号等待'][metric]
            report_content.append(f"{dataset}: {value:.4f}")
            logger.info(f"{dataset} {metric}: {value:.4f}")
    
    # 计算性能差异
    train_rmse = train_metrics['排队到叫号等待']['RMSE']
    val_rmse = val_metrics['排队到叫号等待']['RMSE']
    test_rmse = test_metrics['排队到叫号等待']['RMSE']
    
    # 计算性能差异百分比
    val_diff = ((val_rmse - train_rmse) / train_rmse) * 100
    test_diff = ((test_rmse - train_rmse) / train_rmse) * 100
    
    report_content.append("\n性能差异:")
    report_content.append(f"验证集vs训练集 RMSE增加: {val_diff:.2f}%")
    report_content.append(f"测试集vs训练集 RMSE增加: {test_diff:.2f}%")
    
    # 判断过拟合程度
    if val_diff > 10 or test_diff > 10:
        overfitting_status = "警告：可能存在显著过拟合"
    elif val_diff > 5 or test_diff > 5:
        overfitting_status = "注意：可能存在轻微过拟合"
    else:
        overfitting_status = "模型拟合程度适中"
    
    report_content.append(f"\n拟合状态: {overfitting_status}")
    
    # 保存评估报告
    with open(report_dir / 'overfitting_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"\n过拟合分析报告已保存至: {report_dir / 'overfitting_analysis.txt'}")
    logger.info(f"\n拟合状态: {overfitting_status}")

def train_model():
    """训练模型的主函数"""
    # 创建带时间戳的运行文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Config.MODELS_DIR / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始模型训练流程... 运行ID: {timestamp}")
    
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
    
    # 修改评估报告保存位置
    report_dir = run_dir / 'evaluation_reports'
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
    
    # 修改结果和模型保存位置
    joblib.dump(results, run_dir / 'training_results.joblib')
    
    # 保存模型到运行特定的文件夹
    model_dir = run_dir / 'trained_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'queue_predictor.joblib')
    
    # 添加过拟合检查和学习曲线分析
    check_overfitting(model, 
                     X_train_scaled, y_train,
                     X_val_scaled, y_val,
                     X_test_scaled, y_test,
                     run_dir)
    
    plot_learning_curves(model, X_train_scaled, y_train, X_val_scaled, y_val, run_dir)
    
    logger.info(f"训练完成！模型和结果已保存到: {run_dir}")
    
    return model, results

def train_with_cv(X, y, n_splits=5):
    """使用交叉验证训练模型"""
    # 创建带时间戳的运行文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Config.MODELS_DIR / f'run_{timestamp}_cv'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = XGBoostQueuePredictor()
        model.fit_with_early_stopping(X_train, y_train, X_val, y_val)
        
        val_metrics = model.evaluate(X_val, y_val)
        cv_scores.append(val_metrics)
        
        # 修改评估报告保存位置
        report_dir = run_dir / 'evaluation_reports'
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / f'fold_{fold}_evaluation_report.txt', 'w') as f:
            f.write(f"Fold {fold} Validation Metrics:\n{val_metrics}\n")
        
        # 修改模型保存位置
        model_dir = run_dir / 'trained_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f'fold_{fold}_queue_predictor.joblib')
        
        logger.info(f"\nFold {fold} 验证集评估结果:")
        model.log_metrics(val_metrics, prefix=f'Fold {fold}')
    
    return cv_scores

if __name__ == "__main__":
    train_model()