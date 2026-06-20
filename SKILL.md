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
5. 字幕生成 (ASS格式, 汉仪中黑体加粗 54号, 重点词60号黄色)
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
| FONT_SIZE | 54 | 正文字号（汉仪中黑体加粗） |
| FONT_SIZE_HIGHLIGHT | 60 | 重点词字号 |
| FONT_NAME | HYZhongHei 197 | 字体名称 |
| SILENCE_THRESHOLD | 1.0 | 沉默裁剪阈值(秒) |
| KEY_TERMS | [...] | 自动高亮的重点词列表 |

## 自更新

告诉 AI 你的偏好，例如:
- "字号改成 32"
- "沉默阈值改成 1.5 秒"
- "重点词加 xxx"

AI 会记录并应用到下次处理。

## 🪤 已知陷阱（每次调用前先读）

### FFmpeg 路径转义（Windows 特有）

```
问题:    C:\ 中的冒号(:)在 FFmpeg filter 语法中是选项分隔符
症状:    ass='C:/path/file.ass' → "Unable to parse original_size"
原因:    Git Bash (MSYS2) 自动转换路径；FFmpeg 把 C: 解析为选项
修复:
        1. 从 Python subprocess 调用时无问题（不经过 bash）
        2. 从 bash 调用时:
           - 复制 ASS 到 C:\temp\ 简化路径
           - 用 MSYS2_ARG_CONV_EXCL="*" 禁止路径转换
           - filter 用 ass='C\:/temp/file.ass':original_size=WxH
           见 test_41s.py / test_portrait.py 中的实现
```

### ASS 样式必须是 Bold=1

```
问题:    文字一会儿粗一会儿细
症状:    含数字/关键词的句子加粗，其他句子变细
原因:    Style 定义中 Bold=0，只有命中 KEY_TERMS/NUM_PATTERN 的文本
         才有内联 \b1 标签，纯文本沿用 Style 默认不加粗
修复:    Style 第 8 个字段设为 1（Bold=1）
         同时内联 tags_close 也要用 \b1 不能用 \b0
```

### 禁止使用 \rStyle 标签

```
问题:    \rHighlight\rDefault 渲染为 {} 大括号，文字消失
原因:    DirectWrite/libass 对 \rStyle 支持不完整
修复:    全程使用内联 {\fs\c\b} 标签
         高亮: {\fs60\c&H0000FFFF&\b1}词{\fs54\c&H00FFFFFF&\b1}
```

### DirectWrite 字体名必须精确

```
问题:    指定"汉仪中黑体"后 FFmpeg 找不到字体
原因:    Windows DirectWrite 需要注册在系统的完整字体名
修复:    用 fontTools 查 nameID 1/4/6 获取精确名
         已知可用的名字:
         - HYZhongHei 197（汉仪中黑体，已预装）
         - SimHei（标准黑体，已预装）
         - AaHouDiHei（Aa厚底黑，需 AddFontResource 注册）
         - zihunjingdianrunhei（字魂经典润黑，需注册）
```

### 注册字体要用 AddFontResource

```
问题:    复制 .ttf 到 C:\Windows\Fonts 后 DirectWrite 依然不识别
原因:    文件复制不触发系统字体注册
修复:    ctypes.windll.gdi32.AddFontResourceW(font_path)
        返回 >0 表示注册成功
```

### GBK 编码错误（Windows 特有）

```
问题:    Python 打印中文时报 UnicodeEncodeError
症状:    'gbk' codec can't encode character
修复:    PYTHONIOENCODING=utf-8 python script.py
        subprocess 调用加 capture_output=True, text=True
```

### Whisper 中文语气词检测

```
问题:    "就是"中的"就"被误判为语气词，"三个雷"被裁剪
原因:    Whisper 中文 word-level 是单字粒度；子串匹配误伤内容
修复:    只检测单字语气词（嗯啊呃哦哎诶呀嘛啦哈嚯）
         不做子串包含匹配
```

### 跨段沉默检测

```
问题:    Whisper 段 N 末尾到段 N+1 开头之间的沉默未被检测
原因:    原算法只检查同一段内的 word gap
修复:    增加 across-segment 循环: 段 N 最后词 → 段 N+1 第一词
```

### 调字幕视觉的致命效率问题

```
问题:    调字体/字号/颜色反复 20+ 轮，每次要等 FFmpeg 渲染
教训:    不要"猜→渲染→看→改"循环
        应该先锁死视觉规范表，一次写入，一次验证
规范表: 见 memory/video-subtitle-规范.md

