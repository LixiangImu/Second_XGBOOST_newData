import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import Config
from predict import QueuePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.predictor = QueuePredictor()
    
    def evaluate_model(self, X_test, y_test, n_samples=10):
        """评估模型性能并展示样例"""
        results = []
        
        # 对每条数据进行预测
        for idx, row in X_test.iterrows():
            queue_time = f"{row['排队日期']} {row['排队时间']}"
            pred = self.predictor.predict_waiting_time(queue_time, row['煤种编号'])
            
            results.append({
                '提煤单号': row['提煤单号'] if '提煤单号' in row else f'样本_{idx}',
                '煤种编号': row['煤种编号'],
                '排队时间': queue_time,
                '预测排队到叫号等待': pred['预计排队到叫号等待(分钟)'],
                '预测叫号到入口等待': pred['预计叫号到入口等待(分钟)'],
                '预测总等待时间': pred['预计总等待时间(分钟)'],
                '实际排队到叫号等待': y_test.iloc[idx, 0],  # 第一列是排队到叫号等待
                '实际叫号到入口等待': y_test.iloc[idx, 1],  # 第二列是叫号到入口等待
                '实际总等待时间': y_test.iloc[idx, 0] + y_test.iloc[idx, 1]  # 总等待时间
            })
        
        results_df = pd.DataFrame(results)
        
        # 计算评估指标
        metrics = self._calculate_metrics(results_df)
        
        # 添加预测偏差
        results_df['排队到叫号偏差'] = results_df['预测排队到叫号等待'] - results_df['实际排队到叫号等待']
        results_df['叫号到入口偏差'] = results_df['预测叫号到入口等待'] - results_df['实际叫号到入口等待']
        results_df['总等待时间偏差'] = results_df['预测总等待时间'] - results_df['实际总等待时间']
        
        # 展示评估结果
        self._display_evaluation_results(metrics, results_df, n_samples)
        
        return metrics, results_df
    
    def _calculate_metrics(self, results):
        """计算评估指标"""
        metrics = {}
        
        for wait_type in ['排队到叫号等待', '叫号到入口等待', '总等待时间']:
            pred_col = f'预测{wait_type}'
            actual_col = f'实际{wait_type}'
            
            metrics[f'{wait_type}_mse'] = mean_squared_error(
                results[actual_col], results[pred_col]
            )
            metrics[f'{wait_type}_rmse'] = np.sqrt(metrics[f'{wait_type}_mse'])
            metrics[f'{wait_type}_mae'] = mean_absolute_error(
                results[actual_col], results[pred_col]
            )
            metrics[f'{wait_type}_r2'] = r2_score(
                results[actual_col], results[pred_col]
            )
            
            # 添加平均偏差百分比
            metrics[f'{wait_type}_mape'] = np.mean(
                np.abs(results[f'{pred_col}'] - results[f'{actual_col}']) / 
                (results[f'{actual_col}'] + 1e-8) * 100  # 添加小值避免除零
            )
        
        return metrics
    
    def _display_evaluation_results(self, metrics, results_df, n_samples):
        """展示评估结果"""
        # 1. 显示整体评估指标
        logger.info("\n模型评估指标:")
        for wait_type in ['排队到叫号等待', '叫号到入口等待', '总等待时间']:
            logger.info(f"\n{wait_type}:")
            logger.info(f"  RMSE: {metrics[f'{wait_type}_rmse']:.2f}分钟")
            logger.info(f"  MAE: {metrics[f'{wait_type}_mae']:.2f}分钟")
            logger.info(f"  R²: {metrics[f'{wait_type}_r2']:.4f}")
            logger.info(f"  MAPE: {metrics[f'{wait_type}_mape']:.2f}%")
        
        # 2. 展示随机样例
        logger.info("\n预测样例 (随机抽取):")
        sample_results = results_df.sample(n=n_samples, random_state=42)
        
        for _, row in sample_results.iterrows():
            logger.info("\n" + "="*50)
            logger.info(f"提煤单号: {row['提煤单号']}")
            logger.info(f"煤种编号: {row['煤种编号']}")
            logger.info(f"排队时间: {row['排队时间']}")
            logger.info("\n排队到叫号等待:")
            logger.info(f"  预测: {row['预测排队到叫号等待']:.1f}分钟")
            logger.info(f"  实际: {row['实际排队到叫号等待']:.1f}分钟")
            logger.info(f"  偏差: {row['排队到叫号偏差']:.1f}分钟")
            logger.info("\n叫号到入口等待:")
            logger.info(f"  预测: {row['预测叫号到入口等待']:.1f}分钟")
            logger.info(f"  实际: {row['实际叫号到入口等待']:.1f}分钟")
            logger.info(f"  偏差: {row['叫号到入口偏差']:.1f}分钟")
            logger.info("\n总等待时间:")
            logger.info(f"  预测: {row['预测总等待时间']:.1f}分钟")
            logger.info(f"  实际: {row['实际总等待时间']:.1f}分钟")
            logger.info(f"  偏差: {row['总等待时间偏差']:.1f}分钟")

def main():
    """主函数"""
    try:
        # 打印当前工作目录和数据目录
        logger.info(f"当前工作目录: {Path.cwd()}")
        logger.info(f"数据目录: {Config.PROCESSED_DATA_DIR}")
        
        # 加载测试数据
        X_test = pd.read_csv(Config.PROCESSED_DATA_DIR / 'X_test.csv')
        y_test = pd.read_csv(Config.PROCESSED_DATA_DIR / 'y_test.csv')
        
        logger.info(f"加载测试数据 X: {X_test.shape}, y: {y_test.shape}")
        
        # 创建评估器
        evaluator = ModelEvaluator()
        
        # 评估模型
        metrics, results = evaluator.evaluate_model(X_test, y_test, n_samples=5)
        
        # 保存详细结果
        results.to_csv(Config.MODELS_DIR / 'evaluation_results.csv', index=False)
        logger.info("\n评估结果已保存至 evaluation_results.csv")
        
    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()