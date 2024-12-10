# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """模型基类，定义了所有模型都需要实现的接口"""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测"""
        pass
    
    def evaluate(self, X, y_true):
        """评估模型性能"""
        y_pred = self.predict(X)
        metrics = {}
        
        # 计算每个目标变量的指标
        for i, col in enumerate(['排队到叫号等待', '叫号到入口等待']):
            y_true_i = y_true.iloc[:, i]
            y_pred_i = y_pred[:, i]
            
            metrics[f'{col}_mse'] = mean_squared_error(y_true_i, y_pred_i)
            metrics[f'{col}_rmse'] = np.sqrt(metrics[f'{col}_mse'])
            metrics[f'{col}_mae'] = mean_absolute_error(y_true_i, y_pred_i)
            metrics[f'{col}_r2'] = r2_score(y_true_i, y_pred_i)
            
        return metrics
    
    def log_metrics(self, metrics, prefix=''):
        """记录评估指标"""
        logger.info(f"\n{prefix} Metrics for {self.name}:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")