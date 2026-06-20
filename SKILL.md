---
name: video-agent
description: 口播视频全自动剪辑 Agent。检测沉默(>1s)、语气词(嗯啊呃)、重复内容；背景模糊+人物锐化；ASS字幕（智能换行、重点词高亮）。触发词：剪视频、口播剪辑、视频处理
---

# 口播视频剪辑 Agent

> 基于 OpenCV + FFmpeg + Whisper 的口播视频自动处理流水线。

## 快速使用

```
用户: 帮我剪这个口播视频
用户: 处理视频 original.mp4
```

## 流程

```
1. 分析 → 沉默检测(>1s) + 语气词(嗯啊呃) + 重复内容
2. 提取音频
3. 背景模糊 (OpenCV 人脸检测 + 双重高斯模糊 + 人物锐化)
4. 裁剪 (concat demuxer)
5. 字幕生成 (ASS格式, 汉仪中黑体 28号, 重点词32号黄色)
6. 合成 + 烧录字幕
```

## 执行步骤

### 1. 运行 Sub-Agent

```bash
cd /c/Users/Administrator/Documents/trae_projects/first\ cc/video-project
PYTHONIOENCODING=utf-8 python sub_agent.py
```

### 2. 查看输出

输出文件: `output/final.mp4`

### 3. 参数调整

如需调整字号、模糊强度、沉默阈值等，编辑 `sub_agent.py` 顶部配置区域。

## 输出目录结构

```
output/
├── final.mp4           # 最终成品
temp/
├── blurred.mp4         # 模糊后视频
├── subtitles.ass       # ASS字幕文件
├── tv.mp4 / ta.aac     # 裁剪后的音视频
└── muxed.mp4           # 合成后(字幕前)
```

## 依赖

| 依赖 | 用途 |
|------|------|
| Python 3.8+ | 主运行环境 |
| OpenCV | 人脸检测、图像处理 |
| FFmpeg | 音视频裁剪、字幕烧录 |
| faster-whisper | 语音识别(预装) |
| numpy | 矩阵运算 |

## 配置项

`sub_agent.py` 顶部可调：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| FONT_SIZE | 28 | 正文字号 |
| FONT_SIZE_HIGHLIGHT | 32 | 重点词字号 |
| FONT_NAME | 汉仪中黑体 | 字体名称 |
| SILENCE_THRESHOLD | 1.0 | 沉默裁剪阈值(秒) |
| KEY_TERMS | [...] | 自动高亮的重点词列表 |

## 自更新

告诉 AI 你的偏好，例如:
- "字号改成 32"
- "沉默阈值改成 1.5 秒"
- "重点词加 xxx"

AI 会记录并应用到下次处理。
