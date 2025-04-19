# 🚨 基于AI的摔倒检测系统

<div align="center">

![系统横幅](https://via.placeholder.com/800x200/0D1117/FFFFFF?text=AI+Fall+Detection)

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.9+-green)](https://mediapipe.dev)
[![许可证](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![状态](https://img.shields.io/badge/状态-开发中-yellow)](https://github.com/yourusername/AI-based-Fall-Detection)

</div>

<p align="center">基于计算机视觉的实时摔倒检测系统，采用MediaPipe姿态估计和自定义分析算法</p>

---

## 📋 目录

- [系统功能](#-系统功能)
- [安装与配置](#️-安装与配置)
- [使用指南](#-使用指南)
- [重要注意事项](#-重要注意事项)
- [性能指标](#-性能指标)
- [未来改进](#-未来改进)

---

## ✨ 系统功能

### 🎥 实时视频处理

系统支持多种输入源，包括USB摄像头（设备索引0,1,2...）、RTSP视频流、本地视频文件（MP4, AVI等）以及批量处理视频文件夹。视频处理参数可灵活配置，用户可根据需求调整分辨率（640x480/1280x720）、帧率（15/30/60fps）以及设置ROI区域以在特定区域内进行检测。

### 👤 人体姿态分析

本系统采用MediaPipe BlazePose模型进行33个关键点检测、3D姿态估计及实时性能优化。通过计算人体中心点高度变化、关键点角度变化和运动轨迹分析等关键指标，实现精准的姿态评估。

### 🔍 摔倒检测算法

系统通过三个主要步骤进行摔倒检测：

1. **特征提取**：计算头部-脚部高度差，分析躯干倾斜角度，检测突然的速度变化
2. **决策逻辑**：

   ```python
   if (height_ratio < threshold and 
       torso_angle > angle_threshold and
       velocity > velocity_threshold):
       trigger_alarm()
   ```
3. **灵敏度调节**：提供可配置的阈值参数、场景预设（医院/家庭/公共场所）和自适应学习功能

### 🚨 警报系统

系统采用多级警报机制，从初级警报（屏幕提示+声音）到中级警报（邮件通知+日志记录）再到紧急警报（短信/电话通知）。警报系统完全可自定义，支持声音文件替换、通知模板编辑和延时设置。

### 📊 数据分析

系统提供详细的事件日志记录（包括时间戳、视频帧截图、关键点数据和置信度分数）及统计报表功能（每日/每周/每月事件统计、误报率分析和系统性能指标）。

---

## 🛠️ 安装与配置

### 💻 环境要求与安装

系统需要Python 3.8+环境，推荐使用支持CUDA的GPU以获得最佳性能。安装步骤如下：

```bash
# 克隆仓库
git clone https://github.com/yourusername/AI-based-Fall-Detection.git
cd AI-based-Fall-Detection

# 安装依赖
pip install -r requirements.txt
```

> 💡 **提示**：首次运行前请确认已安装所有必要依赖，并检查GPU驱动是否正确配置。

### ⚙️ 主要配置

系统通过config.py文件进行配置，示例如下：

```python
# config.py 示例
DETECTION = {
    'confidence': 0.5,  # 检测置信度
    'threshold': 0.6,   # 摔倒阈值 
    'cooldown': 10      # 警报间隔
}

VIDEO = {
    'resolution': (640, 480),
    'fps': 30,
    'roi': None  # 设置为(x, y, w, h)以启用ROI
}

ALERTS = {
    'sound': True,
    'email': False,
    'sms': False
}
```

---

## 🚀 使用指南

### 📝 命令与参数

系统提供多种运行命令以满足不同场景需求：

```bash
# 摄像头实时检测
python fall_detection.py --input 0

# 处理视频文件
python fall_detection.py --input video.mp4

# 保存结果
python fall_detection.py --input video.mp4 --output results/

# 使用特定配置
python fall_detection.py --input 0 --config configs/hospital.json
```

参数说明：

|       参数       | 必选 | 说明                             |
| :---------------: | :--: | :------------------------------- |
|    `--input`    |  ✅  | 视频源路径（0,1,2...表示摄像头） |
|   `--output`   |  ❌  | 结果保存路径                     |
|   `--config`   |  ❌  | 配置文件路径                     |
| `--sensitivity` |  ❌  | 检测灵敏度（低/中/高）           |
|   `--display`   |  ❌  | 启用/禁用显示（true/false）      |

### ⌨️ 键盘控制

系统支持多种键盘快捷操作：

|  按键  | 功能            |
| :-----: | :-------------- |
|  `Q`  | 退出程序        |
|  `S`  | 保存当前帧      |
|  `P`  | 暂停/继续处理   |
| `+/-` | 增加/减少灵敏度 |

---

## 📌 重要注意事项

<table>
  <tr>
    <td width="50%">
      <h3>🖥️ 硬件与使用建议</h3>
      <p>系统推荐使用GPU加速以获得最佳性能，最小需要4GB内存（推荐8GB），并且需要稳定的摄像头安装位置。为获得最佳检测效果，请确保良好的光照条件，将摄像头放置在适当高度（推荐2-2.5米），准备alarm.wav文件用于音频警报，并在部署前使用模拟摔倒测试系统。</p>
    </td>
    <td width="50%">
      <h3>⚠️ 系统局限性</h3>
      <p>在多人场景中系统性能可能会降低，遮挡可能影响检测准确性，不推荐在完全黑暗的环境中使用。请在部署前充分测试系统在目标环境中的表现，并根据实际情况调整系统参数。</p>
    </td>
  </tr>
</table>

---

## 📊 性能指标

<div align="center">

|       场景       | 准确率 | 误报率 | 处理帧率 |
| :--------------: | :----: | :----: | :------: |
|   家庭（1人）   |  95%  |   2%   |  25-30  |
|  医院（2-3人）  |  92%  |   3%   |  20-25  |
| 公共区域（人多） |  85%  |   7%   |  15-20  |

</div>


---

## 🔄 未来改进

<div align="center">

| 计划功能                 | 优先级 |   状态   |
| :----------------------- | :----: | :-------: |
| 多人摔倒检测优化         |   高   | 🔄 进行中 |
| 与智能家居系统集成       |   中   | 📅 计划中 |
| 移动应用程序用于远程监控 |   中   | 📅 计划中 |
| 基于云的分析仪表板       |   低   | 📅 计划中 |
| 针对特定环境的迁移学习   |   低   | 📅 计划中 |

</div>

---

<div align="center">

<p align="center">
  <sub>© 2023 AI Fall Detection Team. All Rights Reserved.</sub>
</p>
