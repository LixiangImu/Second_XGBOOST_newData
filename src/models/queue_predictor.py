# src/models/queue_predictor.py
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from .base_model import BaseModel
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class XGBoostQueuePredictor(BaseModel):
    """基于XGBoost的队列等待时间预测模型"""
    
    def __init__(self, params=None):
        super().__init__(name="XGBoostQueuePredictor")
        # 修改基础参数以减少过拟合
        self.base_params = {
            '排队到叫号等待': {
                'objective': 'reg:squarederror',
                'max_depth': 6,             # 降低树的深度
                'learning_rate': 0.01,      # 降低学习率
                'n_estimators': 100,        # 减少树的数量
                'subsample': 0.7,           # 降低样本采样比例
                'colsample_bytree': 0.6,    # 降低特征采样比例
                'min_child_weight': 3,      # 增加最小子节点权重
                'reg_alpha': 0.1,           # 添加L1正则化
                'reg_lambda': 1.0,          # 添加L2正则化
                'random_state': 42
            }
        }
        # 更新用户提供的参数
        if params:
            self.base_params.update(params)
            
        # 创建基础模型
        self.base_model = xgb.XGBRegressor(**self.base_params['排队到叫号等待'])
        self.model = MultiOutputRegressor(self.base_model)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        logger.info(f"\n{'='*50}")
        logger.info("开始训练XGBoost模型...")
        logger.info(f"训练集形状: X_train{X_train.shape}, y_train{y_train.shape}")
        if X_val is not None:
            logger.info(f"验证集形状: X_val{X_val.shape}, y_val{y_val.shape}")
        
        # 记录特征名称
        self.feature_names = X_train.columns.tolist()
        logger.info(f"\n使用的特征 ({len(self.feature_names)}):")
        for i, feat in enumerate(self.feature_names, 1):
            logger.info(f"{i}. {feat}")
        
        # 训练模型
        logger.info("\n开始训练多输出模型...")
        self.model.fit(X_train, y_train)
        
        # 训练后输出每个目标的特征重要性
        for i, target in enumerate(['排队到叫号等待']):
            logger.info(f"\n{target} - 特征重要性:")
            importance = self.model.estimators_[i].feature_importances_
            feat_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            })
            feat_imp = feat_imp.sort_values('importance', ascending=False)
            
            logger.info(f"\n{target} - 特征重要性 Top 10:")
            for _, row in feat_imp.head(10).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        # 评估训练集性能
        train_metrics = self.evaluate(X_train, y_train)
        self.log_metrics(train_metrics, prefix='Training')
        
        # 如果有验证集，评估验证集性能
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.log_metrics(val_metrics, prefix='Validation')
            
        logger.info(f"\n{'='*50}")
        return train_metrics
    
    def fit_with_early_stopping(self, X_train, y_train, X_val, y_val):
        """使用早停的训练方法"""
        self.model = []
        for i, target in enumerate(['排队到叫号等待']):
            model = xgb.XGBRegressor(
                **self.base_params[target],
                early_stopping_rounds=10,
                eval_metric=['rmse', 'mae']
            )
            eval_set = [(X_train, y_train.iloc[:, i]), (X_val, y_val.iloc[:, i])]
            model.fit(
                X_train, y_train.iloc[:, i],
                eval_set=eval_set,
                verbose=True
            )
            self.model.append(model)
        
    def predict(self, X):
        """预测"""
        if isinstance(self.model, list):
            # 如果是使用早停训练的模型列表
            predictions = np.column_stack([
                model.predict(X) for model in self.model
            ])
        else:
            # 如果是MultiOutputRegressor
            predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self):
        """获取特征重要性"""
        importance_dict = {}
        feature_names = self.feature_names if hasattr(self, 'feature_names') else None
        
        if isinstance(self.model, list):
            # 如果是使用早停训练的模型列表
            for i, model in enumerate(self.model):
                importance = model.feature_importances_
                target = ['排队到叫号等待'][i]
                
                if feature_names:
                    importance_dict[target] = dict(zip(feature_names, importance))
                else:
                    importance_dict[target] = dict(enumerate(importance))
        else:
            # 如果是MultiOutputRegressor
            for i, estimator in enumerate(self.model.estimators_):
                importance = estimator.feature_importances_
                target = ['排队到叫号等待'][i]
                
                if feature_names:
                    importance_dict[target] = dict(zip(feature_names, importance))
                else:
                    importance_dict[target] = dict(enumerate(importance))
        
        return importance_dict
    
    def log_feature_importance(self):
        """记录特征重要性"""
        importance_dict = self.get_feature_importance()
        
        logger.info("\n特征重要性分析:")
        for target, importance in importance_dict.items():
            logger.info(f"\n{target} - Top 10 特征:")
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
            for feature, imp in list(sorted_importance.items())[:10]:
                logger.info(f"{feature}: {imp:.4f}")