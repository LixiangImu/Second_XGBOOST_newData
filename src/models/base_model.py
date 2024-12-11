# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import pandas as pd

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
        
        # 确保y_true是DataFrame
        if not isinstance(y_true, pd.DataFrame):
            y_true = pd.DataFrame(y_true, columns=['排队到叫号等待'])
        
        # 计算指标
        metrics = {}
        target = '排队到叫号等待'
        
        # 获取预测值和真实值
        if isinstance(y_pred, np.ndarray) and len(y_pred.shape) > 1:
            y_pred_i = y_pred[:, 0]
        else:
            y_pred_i = y_pred
            
        y_true_i = y_true[target] if isinstance(y_true, pd.DataFrame) else y_true
        
        # 计算各项指标
        mse = mean_squared_error(y_true_i, y_pred_i)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_i, y_pred_i)
        r2 = r2_score(y_true_i, y_pred_i)
        
        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    def log_metrics(self, metrics, prefix=''):
        """记录评估指标"""
        logger.info(f"\n{prefix} 指标:")
        for target, target_metrics in metrics.items():
            logger.info(f"\n{target}:")
            for metric_name, value in target_metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")