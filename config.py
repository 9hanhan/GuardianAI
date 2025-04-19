import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检测参数
DETECTION_CONFIG = {
    'min_detection_confidence': 0.5,    # 降低检测置信度要求
    'min_tracking_confidence': 0.5,     # 降低跟踪置信度要求
    'fall_threshold': 0.4,              # 调整摔倒阈值，使其更容易触发
    'alert_cooldown': 2,                # 减少警报冷却时间
}

# 视频处理参数
VIDEO_CONFIG = {
    'resize_width': 640,    # 降低分辨率以提高处理速度
    'resize_height': 480,   # 降低分辨率以提高处理速度
    'fps': 30,
}

# 通知配置
NOTIFICATION_CONFIG = {
    'enable_email': False,
    'smtp_server': os.getenv('SMTP_SERVER', ''),
    'smtp_port': int(os.getenv('SMTP_PORT', 587)),
    'smtp_username': os.getenv('SMTP_USERNAME', ''),
    'smtp_password': os.getenv('SMTP_PASSWORD', ''),
    'notification_email': os.getenv('NOTIFICATION_EMAIL', ''),
}

# 日志配置
LOG_CONFIG = {
    'log_file': 'fall_detection.log',
    'log_level': 'INFO',
}

# 摄像头配置
CAMERA_CONFIG = {
    'camera_id': 0,  # 默认摄像头ID
    'use_webcam': False,  # 是否使用网络摄像头
} 