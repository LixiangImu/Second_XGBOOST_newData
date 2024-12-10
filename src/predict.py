import pandas as pd
import joblib
from datetime import datetime
import logging
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueuePredictor:
    def __init__(self):
        self.model = joblib.load(Config.MODELS_DIR / 'queue_predictor.joblib')
        self.scaler = joblib.load(Config.MODELS_DIR / 'scaler.joblib')
        self.feature_engineer = joblib.load(Config.MODELS_DIR / 'feature_engineer.joblib')
    
    def prepare_single_prediction(self, queue_time, coal_type):
        """准备单条预测数据
        
        Args:
            queue_time (str): 排队时间，格式："YYYY-MM-DD HH:MM"
            coal_type (str): 煤种编号，例如："YT02WX"
            
        Returns:
            pd.DataFrame: 准备好的特征数据
        """
        # 解析时间
        dt = datetime.strptime(queue_time, "%Y-%m-%d %H:%M")
        
        # 创建基础数据
        data = pd.DataFrame([{
            '排队日期': dt.date(),
            '排队时间': dt.strftime("%H:%M"),
            '排队小时': dt.hour,
            '排队分钟': dt.minute,
            '煤种编号': coal_type,
            '星期': dt.weekday(),
            '是否周末': 1 if dt.weekday() >= 5 else 0
        }])
        
        # 使用特征工程器处理数据
        features = self.feature_engineer.process_single(data)
        
        # 标准化特征
        numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
        features[numeric_features] = self.scaler.transform(features[numeric_features])
        
        return features
    
    def predict_waiting_time(self, queue_time, coal_type):
        """预测单辆车的等待时间
        
        Args:
            queue_time (str): 排队时间，格式："YYYY-MM-DD HH:MM"
            coal_type (str): 煤种编号
            
        Returns:
            dict: 预测的等待时间
        """
        try:
            # 准备特征
            features = self.prepare_single_prediction(queue_time, coal_type)
            
            # 进行预测
            predictions = self.model.predict(features)
            
            # 返回预测结果
            return {
                '预计排队到叫号等待(分钟)': round(predictions[0][0], 1),
                '预计叫号到入口等待(分钟)': round(predictions[0][1], 1),
                '预计总等待时间(分钟)': round(sum(predictions[0]), 1)
            }
            
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}", exc_info=True)
            raise

def main():
    """示例使用"""
    try:
        # 创建预测器
        predictor = QueuePredictor()
        
        # 示例预测
        queue_time = "2024-10-16 08:30"  # 排队时间
        coal_type = "YT02WX"             # 煤种编号
        
        # 进行预测
        result = predictor.predict_waiting_time(queue_time, coal_type)
        
        # 打印预测结果
        logger.info("\n预测结果:")
        for key, value in result.items():
            logger.info(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()