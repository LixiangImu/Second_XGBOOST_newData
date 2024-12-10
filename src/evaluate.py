import logging
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import Config
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.results_dir = Config.MODELS_DIR / 'evaluation_results'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_data(self):
        """加载模型和数据"""
        try:
            # 加载模型和标准化器
            self.model = joblib.load(Config.MODELS_DIR / 'queue_predictor.joblib')
            self.scaler = joblib.load(Config.MODELS_DIR / 'scaler.joblib')
            
            # 加载原始测试数据
            X_test = pd.read_csv(Config.DATA_DIR / 'processed' / 'X_test.csv')
            y_test = pd.read_csv(Config.DATA_DIR / 'processed' / 'y_test.csv')
            
            # 加载标签编码器
            self.feature_engineer.label_encoders = joblib.load(
                Config.MODELS_DIR / 'label_encoders.joblib'
            )
            
            # 应用特征工程
            X_test_processed = self.feature_engineer.process(X_test)
            
            # 应用特征缩放
            feature_names = self.scaler.feature_names_in_
            X_test_scaled = X_test_processed.copy()
            X_test_scaled[feature_names] = self.scaler.transform(X_test_processed[feature_names])
            
            logger.info("模型和数据加载成功")
            logger.info(f"特征列表: {feature_names.tolist()}")
            
            return X_test_scaled, y_test, X_test
            
        except Exception as e:
            logger.error(f"加载模型或数据时出错: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        metrics = {}
        target_names = ['排队到叫号等待', '叫号到入口等待']
        
        for i, target in enumerate(target_names):
            metrics[target] = {
                'MSE': mean_squared_error(y_true.iloc[:, i], y_pred[:, i]),
                'RMSE': np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])),
                'MAE': mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
                'R2': r2_score(y_true.iloc[:, i], y_pred[:, i]),
                'MAPE': np.mean(np.abs((y_true.iloc[:, i] - y_pred[:, i]) / y_true.iloc[:, i])) * 100
            }
        
        return metrics
    
    def plot_prediction_scatter(self, y_true, y_pred):
        """绘制预测值与真实值的散点图"""
        target_names = ['排队到叫号等待', '叫号到入口等待']
        
        plt.figure(figsize=(15, 6))
        for i, target in enumerate(target_names):
            plt.subplot(1, 2, i+1)
            plt.scatter(y_true.iloc[:, i], y_pred[:, i], alpha=0.5)
            plt.plot([y_true.iloc[:, i].min(), y_true.iloc[:, i].max()], 
                    [y_true.iloc[:, i].min(), y_true.iloc[:, i].max()], 
                    'r--', lw=2)
            plt.xlabel('实际等待时间 (分钟)')
            plt.ylabel('预测等待时间 (分钟)')
            plt.title(f'{target}预测散点图')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_scatter.png')
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred):
        """绘制预测误差分布图"""
        target_names = ['排队到叫号等待', '叫号到入口等待']
        errors = y_pred - y_true.values
        
        plt.figure(figsize=(15, 6))
        for i, target in enumerate(target_names):
            plt.subplot(1, 2, i+1)
            sns.histplot(errors[:, i], kde=True)
            plt.xlabel('预测误差 (分钟)')
            plt.ylabel('频率')
            plt.title(f'{target}预测误差分布')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'error_distribution.png')
        plt.close()
    
    def analyze_feature_importance(self, X_test):
        """分析特征重要性"""
        importance_dict = self.model.get_feature_importance()
        
        plt.figure(figsize=(15, 10))
        for i, (target, importance) in enumerate(importance_dict.items()):
            plt.subplot(2, 1, i+1)
            
            # 获取前15个最重要的特征
            sorted_importance = dict(sorted(importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:15])
            
            # 创建水平条形图
            plt.barh(list(sorted_importance.keys()), 
                    list(sorted_importance.values()))
            plt.title(f'{target} - 特征重要性 Top 15')
            plt.xlabel('重要性得分')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png')
        plt.close()
    
    def evaluate(self):
        """执行完整的模型评估"""
        try:
            logger.info("开始模型评估...")
            
            # 加载数据和模型
            X_test_scaled, y_test, X_test = self.load_model_and_data()
            
            # 进行预测
            y_pred = self.model.predict(X_test_scaled)
            
            # 计算评估指标
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # 记录评估结果
            logger.info("\n模型评估指标:")
            for target, target_metrics in metrics.items():
                logger.info(f"\n{target}:")
                for metric_name, value in target_metrics.items():
                    logger.info(f"{metric_name}: {value:.4f}")
            
            # 生成可视化
            logger.info("\n生成评估可视化...")
            self.plot_prediction_scatter(y_test, y_pred)
            self.plot_error_distribution(y_test, y_pred)
            self.analyze_feature_importance(X_test_scaled)
            
            # 保存评估结果
            evaluation_results = {
                'metrics': metrics,
                'predictions': y_pred,
                'true_values': y_test.values,
                'feature_names': X_test_scaled.columns.tolist()
            }
            joblib.dump(evaluation_results, 
                       self.results_dir / 'evaluation_results.joblib')
            
            logger.info(f"\n评估完成！结果已保存至: {self.results_dir}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"评估过程中出错: {str(e)}")
            raise

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()