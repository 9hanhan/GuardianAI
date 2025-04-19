import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
from config import LOG_CONFIG, NOTIFICATION_CONFIG
import codecs

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('FallDetection')
        self.logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set up file handler with GBK encoding
        log_file = os.path.join(log_dir, 'fall_detection.log')
        file_handler = logging.FileHandler(log_file, mode='a', encoding='gbk')
        file_handler.setLevel(logging.INFO)
        
        # Set up console handler with UTF-8
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Force immediate flush
        file_handler.terminator = '\n'
        file_handler.flush()
        
        # Set log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Remove all existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_fall(self, video_path, timestamp):
        message = f"Fall detected - {timestamp}"
        self.logger.warning(message)
        return message

class Notification:
    def __init__(self):
        self.config = NOTIFICATION_CONFIG
        self.logger = Logger()
    
    def send_email(self, subject, message):
        if not self.config['enable_email']:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['smtp_username']
            msg['To'] = self.config['notification_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_username'], self.config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.logger.info("Email notification sent successfully")
            return True
        except Exception as e:
            self.logger.logger.error(f"Email notification sending failed: {str(e)}")
            return False
    
    def notify_fall(self, video_path, timestamp):
        if not self.config['enable_email']:
            return
        
        subject = "Fall Detection Alert"
        message = f"""
        Fall detected!
        
        Details:
        - Video path: {video_path}
        - Detection time: {timestamp}
        - System time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_email(subject, message)
