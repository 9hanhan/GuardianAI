import cv2
import numpy as np
import mediapipe as mp
import os
import time
import pygame
import threading
import argparse
from datetime import datetime
from config import DETECTION_CONFIG, VIDEO_CONFIG, CAMERA_CONFIG
from utils import Logger, Notification

class FallDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=DETECTION_CONFIG['min_detection_confidence'],
            min_tracking_confidence=DETECTION_CONFIG['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.fall_threshold = DETECTION_CONFIG['fall_threshold']
        self.alarm_sound = "报警声.mp3"
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # 警报冷却时间（秒）
        self.logger = Logger()
        self.notifier = Notification()
        self.current_video_path = None  # 添加当前视频路径属性
        
        # 初始化pygame音频系统
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"无法初始化音频系统: {str(e)}")
        
        # 添加状态跟踪
        self.prev_landmarks = None
        self.motion_history = []
        self.fall_detected_frames = 0
        self.required_fall_frames = 3   # 增加连续帧数要求，减少误报
        self.min_frames_between_falls = 15  # 增加两次摔倒事件之间的间隔
        self.frames_since_last_fall = 0  # 距离上次摔倒的帧数
        self.debug_info = {}
        self.alarm_playing = False
        self.alarm_played_for_current_fall = False  # 跟踪当前摔倒是否已播放声音
        self.alert_window = None
        self.alert_start_time = 0
        self.alert_duration = 0.5  # 独立警报窗口显示时间（秒）
        self.alert_in_video_duration = 0.5  # 视频内警告效果的持续时间（秒）
        self.last_fall_detection_time = 0  # 用于跟踪最后一次摔倒检测的时间
        self.fall_events_count = 0  # 计数检测到的摔倒事件数
        self.last_alert_window_time = 0  # 上次显示警告窗口的时间
        self.velocity_threshold = 0.025  # 调整速度阈值，使检测更准确
        self.height_ratio_threshold = 0.33  # 稍微调整身高比例阈值
        self.pose_history_size = 10  # 姿态历史记录大小
        self.min_conditions_for_fall = 2  # 判定摔倒所需的最小条件数
        self.low_posture_duration = 0  # 跟踪低姿态持续时间
        self.previous_height = None  # 跟踪上一帧的高度
        self.height_history = []  # 跟踪高度变化历史

    def calculate_body_velocity(self, current_landmarks, prev_landmarks):
        if prev_landmarks is None:
            return 0
        
        try:
            velocity = 0
            key_points = [
                self.mp_pose.PoseLandmark.NOSE.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value
            ]
            
            valid_points = 0
            for i in key_points:
                if i < len(current_landmarks) and i < len(prev_landmarks):
                    dx = current_landmarks[i].x - prev_landmarks[i].x
                    dy = current_landmarks[i].y - prev_landmarks[i].y
                    velocity += np.sqrt(dx*dx + dy*dy)
                    valid_points += 1
            
            return velocity / max(valid_points, 1)  # 避免除以零
        except Exception as e:
            print(f"速度计算错误: {str(e)}")
            return 0

    def check_horizontal_position(self, landmarks):
        try:
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            
            # 计算多个躯干部位的角度
            # 躯干角度
            trunk_angle = abs(np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ))
            
            # 髋部角度
            hip_angle = abs(np.arctan2(
                right_hip.y - left_hip.y,
                right_hip.x - left_hip.x
            ))
            
            # 躯干与垂直方向的角度
            upper_body_vertical = abs(np.arctan2(
                right_shoulder.y - right_hip.y,
                right_shoulder.x - right_hip.x
            ))
            
            # 腿部与垂直方向的角度
            leg_vertical = abs(np.arctan2(
                right_knee.y - right_hip.y,
                right_knee.x - right_hip.x
            ))
            
            # 综合多个角度判断水平状态
            horizontal_score = 0
            if trunk_angle < 0.7:  # 约40度
                horizontal_score += 1
            if hip_angle < 0.7:
                horizontal_score += 1
            if abs(upper_body_vertical - np.pi/2) < 0.5:  # 接近水平
                horizontal_score += 1
            if abs(leg_vertical - np.pi/2) < 0.5:  # 腿部接近水平
                horizontal_score += 1
                
            # 需要至少两个条件满足
            return horizontal_score >= 2
        except Exception as e:
            print(f"水平状态检测错误: {str(e)}")
            return False

    def detect_fall(self, landmarks, frame_height):
        if not landmarks:
            return False
            
        try:
            # 1. 计算人体高度比例
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # 计算整体高度和髋部高度
            height = min(abs(nose.y - left_ankle.y), abs(nose.y - right_ankle.y))
            hip_height = min(abs(left_hip.y - left_ankle.y), abs(right_hip.y - right_ankle.y))
            
            # 跟踪高度变化方向
            height_direction = 0  # 默认为无变化
            if self.previous_height is not None:
                # 负值表示高度减小（从高到低），正值表示高度增加（从低到高）
                height_direction = self.previous_height - height
            
            # 更新高度历史
            self.height_history.append(height)
            if len(self.height_history) > 5:  # 保持最近5帧的历史
                self.height_history.pop(0)
            
            # 检查最近的高度变化趋势
            height_trend = 0
            if len(self.height_history) >= 3:
                # 计算最近几帧的高度变化趋势
                height_trend = self.height_history[0] - self.height_history[-1]
            
            # 更新当前高度作为下一帧的previous_height
            self.previous_height = height
            
            # 跟踪低姿态持续时间
            if height < 0.15 or hip_height < 0.1:
                self.low_posture_duration += 1
            else:
                self.low_posture_duration = 0
            
            # 2. 检查水平姿态
            is_horizontal = self.check_horizontal_position(landmarks)
            
            # 3. 计算速度
            current_landmarks = [landmark for landmark in landmarks]
            velocity = self.calculate_body_velocity(current_landmarks, self.prev_landmarks)
            self.prev_landmarks = current_landmarks
            
            # 4. 综合判断
            conditions_met = 0
            
            # 4.1 高度判断 - 考虑整体高度和髋部高度
            if height < self.height_ratio_threshold or hip_height < self.height_ratio_threshold * 0.5:
                conditions_met += 1
            
            # 4.2 水平状态判断
            if is_horizontal:
                conditions_met += 1
            
            # 4.3 速度判断 - 使用动态阈值
            if velocity > self.velocity_threshold:
                conditions_met += 1
                # 如果速度特别大，增加权重
                if velocity > self.velocity_threshold * 2:
                    conditions_met += 1
            
            # 4.4 姿态突变判断
            pose_changed = False
            if len(self.motion_history) >= 3:
                recent_velocities = self.motion_history[-3:]
                if max(recent_velocities) > self.velocity_threshold * 1.5:
                    pose_changed = True
                    conditions_met += 1
            
            # 更新运动历史
            self.motion_history.append(velocity)
            if len(self.motion_history) > self.pose_history_size:
                self.motion_history.pop(0)
            
            # 判断是否摔倒 - 满足指定数量的条件
            is_falling = conditions_met >= self.min_conditions_for_fall
            
            # 特殊情况：针对最后几次可能的摔倒模式（低姿态但速度不快的情况）
            # 修改后的特殊情况判断 - 要求低姿态同时配合速度或姿态变化
            if (height < 0.10 or hip_height < 0.05) and (velocity > 0.01 or pose_changed or is_horizontal):
                # 非常低的姿态且有一定速度、姿态变化或水平状态，可能是摔倒
                is_falling = True
            
            # 长时间低姿态判断 - 如果低姿态持续太久，可能是正常行为而非摔倒
            if self.low_posture_duration > 30 and velocity < self.velocity_threshold * 0.5:
                # 长时间低姿态且无明显运动，认为是正常行为
                is_falling = False
            
            # *** 新增：检查高度变化方向 ***
            # 只有从高到低的变化才被视为可能的摔倒
            if height_trend < 0.03:  # 高度没有明显减小或者是增加的
                is_falling = False
            
            # 如果是从低到高的移动，不触发警告
            if height_direction < 0:  # 高度在增加（从低到高）
                is_falling = False
            
            # 打印详细调试信息到控制台
            print(f"检测值: 高度={height:.4f}, 髋部高度={hip_height:.4f}, 阈值={self.height_ratio_threshold}, "
                  f"是否水平={is_horizontal}, 速度={velocity:.4f}, 姿态突变={pose_changed}, 满足条件={conditions_met}, "
                  f"低姿态持续={self.low_posture_duration}, 高度变化方向={height_direction:.4f}, 高度趋势={height_trend:.4f}")
            print(f"检测结果: {'摔倒' if is_falling else '正常'}")
            
            # 保存调试信息
            self.debug_info = {
                'height': height,
                'hip_height': hip_height,
                'is_horizontal': is_horizontal,
                'velocity': velocity,
                'pose_changed': pose_changed,
                'conditions_met': conditions_met,
                'fall_frames': self.fall_detected_frames,
                'low_duration': self.low_posture_duration,
                'height_dir': height_direction,
                'height_trend': height_trend
            }
            
            return is_falling
            
        except Exception as e:
            print(f"摔倒检测错误: {str(e)}")
            return False

    def draw_debug_info(self, frame):
        y = 30
        for key, value in self.debug_info.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 30

    def show_alert_window(self, message):
        # 创建更小更美观的警报窗口
        alert_height = 150
        alert_width = 400
        alert_frame = np.zeros((alert_height, alert_width, 3), dtype=np.uint8)
        
        # 创建渐变背景
        for y in range(alert_height):
            # 从深红色渐变到浅红色
            red_value = min(255, int(180 + (y / alert_height) * 75))
            blue_value = min(80, int((y / alert_height) * 80))
            alert_frame[y, :] = (blue_value, 40, red_value)
        
        # 添加半透明边框
        border_thickness = 5
        # 上边框
        alert_frame[0:border_thickness, :] = [100, 100, 255]
        # 下边框
        alert_frame[alert_height-border_thickness:alert_height, :] = [100, 100, 255]
        # 左边框
        alert_frame[:, 0:border_thickness] = [100, 100, 255]
        # 右边框
        alert_frame[:, alert_width-border_thickness:alert_width] = [100, 100, 255]
        
        # 添加警告图标
        icon_size = 40
        icon_margin = 20
        # 创建警告三角形
        cv2.rectangle(alert_frame, 
                     (icon_margin, alert_height//2 - icon_size//2),
                     (icon_margin + icon_size, alert_height//2 + icon_size//2),
                     (0, 0, 0), -1)
        # 绘制警告符号
        cv2.putText(alert_frame, "!", 
                  (icon_margin + icon_size//2 - 5, alert_height//2 + 12),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 添加警告文字
        main_text = "FALL DETECTED"
        font_scale = 0.9
        thickness = 2
        text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = icon_margin + icon_size + 20
        text_y = alert_height//2 - 10
        
        # 绘制主文字
        cv2.putText(alert_frame, main_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # 绘制事件编号
        event_text = message.split("(#")[1].replace(")", "") if "(#" in message else ""
        if event_text:
            sub_text = f"Event #{event_text}"
            cv2.putText(alert_frame, sub_text, (text_x, text_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        
        # 当前时间
        time_text = datetime.now().strftime('%H:%M:%S')
        time_text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(alert_frame, time_text, 
                   (alert_width - time_text_size[0] - 10, alert_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
        
        # 显示警报窗口
        cv2.namedWindow('Fall Alert', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fall Alert', alert_width, alert_height)
        # 设置窗口位置在屏幕右上角
        cv2.moveWindow('Fall Alert', 1300, 30)
        cv2.imshow('Fall Alert', alert_frame)
        self.alert_window = alert_frame
        self.alert_start_time = time.time()
        
        # 修改警报持续时间
        self.alert_duration = 3.0  # 增加到3秒，让用户有足够时间注意到警报

    def update_alert_window(self):
        if self.alert_window is not None:
            current_time = time.time()
            if current_time - self.alert_start_time < self.alert_duration:
                cv2.imshow('Fall Alert', self.alert_window)
            else:
                cv2.destroyWindow('Fall Alert')
                self.alert_window = None

    def process_frame(self, frame):
        frame = cv2.resize(frame, (VIDEO_CONFIG['resize_width'], VIDEO_CONFIG['resize_height']))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        fall_detected = False
        current_time = time.time()
        
        # 更新距离上次摔倒的帧数计数
        self.frames_since_last_fall += 1
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_height = frame.shape[0]
            
            # 检测摔倒
            is_falling = self.detect_fall(landmarks, frame_height)
            
            # 使用帧计数器，连续检测到多帧才触发警告
            if is_falling:
                self.fall_detected_frames += 1
            else:
                # 如果不再摔倒，重置帧计数
                self.fall_detected_frames = 0
                # 修改：只在确实检测到过摔倒的情况下才重置警报状态
                if current_time - self.last_fall_detection_time > 3.0 and self.alarm_played_for_current_fall:
                    self.reset_alarm_state()
                    # 重置警告窗口状态
                    self.last_alert_window_time = 0
            
            # 判断是否为新的摔倒事件（需要满足连续帧数和间隔条件）
            is_new_fall = (self.fall_detected_frames >= self.required_fall_frames and 
                          self.frames_since_last_fall > self.min_frames_between_falls)
            
            if is_new_fall:
                fall_detected = True
                self.last_fall_detection_time = current_time
                self.frames_since_last_fall = 0  # 重置帧计数
                self.fall_events_count += 1  # 增加摔倒事件计数
                self.reset_alarm_state()  # 重要: 立即重置警报状态，确保新的摔倒事件可以触发警报
                
                # 打印摔倒事件统计
                print(f"\n*** 检测到第 {self.fall_events_count} 次摔倒事件 ***\n")
                
                # 触发警报 - 确保同一次摔倒事件只触发一次警报
                if not self.alarm_played_for_current_fall:
                    self.trigger_alarm()
                
                # 记录日志
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                video_name = os.path.basename(self.current_video_path) if hasattr(self, 'current_video_path') and self.current_video_path else "test2.mp4"
                self.logger.logger.warning(f"Fall detected event #{self.fall_events_count} - Video: {video_name}, Time: {timestamp}")
                
                # 只在同一个摔倒事件第一帧时显示警告窗口
                # 通过检查距离上次弹窗的时间是否超过了警报冷却时间来判断
                if current_time - self.last_alert_window_time > self.alert_cooldown:
                    self.show_alert_window(f"FALL DETECTED! (#{self.fall_events_count})")
                    self.last_alert_window_time = current_time
            else:
                # 检查是否在警告效果持续时间内
                if current_time - self.last_fall_detection_time < self.alert_in_video_duration:
                    # 仍在警告效果持续时间内，保持警告显示
                    fall_detected = True
                else:
                    fall_detected = False
                    self.stop_alarm()
            
            # 绘制关键点和骨架
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # 如果检测到摔倒或在显示时间内，绘制警告效果
            if fall_detected or current_time - self.last_fall_detection_time < self.alert_in_video_duration:
                self.draw_alert(frame)
            
            # 更新警报窗口
            self.update_alert_window()
            
            # 绘制调试信息
            self.draw_debug_info(frame)
        
        return fall_detected, frame

    def draw_alert(self, frame):
        # 只绘制红色警告框，不添加文字
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

    def trigger_alarm(self):
        # 检查是否已经为当前摔倒播放过声音
        if not self.alarm_playing and not self.alarm_played_for_current_fall:
            self.alarm_playing = True
            self.alarm_played_for_current_fall = True  # 标记为已为当前摔倒播放声音
            print(f"触发警报声音 - 第 {self.fall_events_count} 次摔倒事件")  # 调试信息
            def play_alarm():
                try:
                    # 确保pygame已初始化
                    if not pygame.mixer.get_init():
                        pygame.mixer.init()
                    
                    pygame.mixer.music.load(self.alarm_sound)
                    pygame.mixer.music.play(0)  # 只播放一次(0表示不循环)
                    print("正在播放警报声...")  # 调试信息
                    
                    # 等待声音播放完成或者被停止
                    start_time = time.time()
                    while pygame.mixer.music.get_busy() and self.alarm_playing and time.time() - start_time < 5:  # 最多等待5秒
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"播放警报声失败: {str(e)}")
                    self.logger.logger.error(f"Cannot play alarm sound: {str(e)}")
                finally:
                    self.alarm_playing = False
                    # 不要完全退出mixer, 可能导致下次无法初始化
                    pygame.mixer.music.stop()
            
            alarm_thread = threading.Thread(target=play_alarm)
            alarm_thread.daemon = True  # 设置为守护线程，避免程序退出时线程仍在运行
            alarm_thread.start()

    def stop_alarm(self):
        self.alarm_playing = False
        
    def reset_alarm_state(self):
        # 重置警报状态，允许下次摔倒时再次播放声音
        self.alarm_played_for_current_fall = False
        print("重置警报状态 - 准备下一次摔倒检测")  # 调试信息

def process_video(video_path, output_path=None):
    detector = FallDetector()
    detector.current_video_path = video_path  # 设置当前视频路径
    
    # 判断是视频文件还是摄像头
    if video_path.isdigit() or CAMERA_CONFIG['use_webcam']:
        cap = cv2.VideoCapture(int(video_path) if video_path.isdigit() else CAMERA_CONFIG['camera_id'])
        video_path = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        detector.logger.logger.error(f"无法打开视频源: {video_path}")
        return
    
    width = VIDEO_CONFIG['resize_width']
    height = VIDEO_CONFIG['resize_height']
    fps = VIDEO_CONFIG['fps']
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    fall_detected = False
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            fall_detected, processed_frame = detector.process_frame(frame)
            
            if fall_detected:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                detector.logger.log_fall(video_path, timestamp)
                detector.notifier.notify_fall(video_path, timestamp)
            
            if writer:
                writer.write(processed_frame)
            
            cv2.imshow('Fall Detection', processed_frame)
            
            # 使用视频原始帧率播放
            delay = max(1, int(1000/cap.get(cv2.CAP_PROP_FPS)))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    except Exception as e:
        detector.logger.logger.error(f"处理视频时出错: {str(e)}")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

def process_directory(input_dir, output_dir=None):
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(video_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}") if output_dir else None
            
            print(f"处理视频: {filename}")
            process_video(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='人体摔倒检测系统')
    parser.add_argument('--input', required=True, help='输入视频文件、目录路径或摄像头ID')
    parser.add_argument('--output', help='输出目录路径（可选）')
    parser.add_argument('--webcam', action='store_true', help='使用网络摄像头')
    
    args = parser.parse_args()
    
    if args.webcam:
        CAMERA_CONFIG['use_webcam'] = True
    
    if os.path.isfile(args.input):
        process_video(args.input, args.output)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        process_video(args.input, args.output)
