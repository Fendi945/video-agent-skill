#!/usr/bin/env python3
"""
口播视频剪辑 Agent v1
=======================
专用子 agent，负责口播视频的全自动剪辑。

功能：
1. 检测并裁剪：沉默(>1s)、语气词(嗯啊呃)、重复内容、废话
2. 背景模糊 + 人物锐化 (OpenCV人脸检测)
3. 智能字幕 (ASS格式，汉仪中黑体加粗 54号，重点词60号黄色高亮，智能换行)
"""

import json
import subprocess
import time
import sys
import re
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = Path(__file__).parent
ORIGINAL_VIDEO = PROJECT_DIR / "original.mp4"
TRANSCRIPT_FILE = PROJECT_DIR / "transcript.json"
OUTPUT_DIR = PROJECT_DIR / "output"
TEMP_DIR = PROJECT_DIR / "temp"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# 字体设置
FONT_NAME = "HYZhongHei 197"     # 汉仪中黑体（Windows注册名，已预装）
FONT_FALLBACK = "SimHei"
FONT_SIZE = 54            # 正文字号
FONT_SIZE_HIGHLIGHT = 60  # 重点词字号
FONT_COLOR = "&H00FFFFFF"      # 白色
FONT_COLOR_HIGHLIGHT = "&H0000FFFF"  # 黄色 (ASS: &HAABBGGRR)

# 沉默阈值
SILENCE_THRESHOLD = 1.0  # 超过1秒沉默视为气口

# 语气词列表 (仅确凿无疑的语气词，避免误伤内容词)
FILLER_WORDS = {
    "嗯": 0.3, "啊": 0.3, "呃": 0.3, "哦": 0.3,
    "哎": 0.3, "诶": 0.3, "呀": 0.3, "嘛": 0.3,
    "啦": 0.3, "哈": 0.3, "嚯": 0.3,
}

# Whisper 错字修正词典
CORRECTIONS = {
    "时工钱多": "施工前", "动线每一会儿": "动线没画", "时工对黑": "施工队黑",
    "守五王哪身": "手往哪伸", "红风成一三": "红枫×3", "元宝风成一二": "元宝枫×2",
    "从客厅到水井": "从客厅到水景", "踩一脚衣": "踩一脚泥", "洞个脑子": "动动脑子",
    "叶竹省": "业主审", "怎么省它": "怎么省事", "砸迟毕": "砸池壁",
    "拿枝笔": "拿支笔", "拿着笔": "拿支笔", "循环笵": "循环泵",
    "时工": "施工", "鞋俩字": "写俩字", "绿花袋": "绿化带", "厅步路": "汀步路",
    "红风": "红枫", "元宝风": "元宝枫", "贯笵": "冠幅", "笵坑": "泵坑",
    "迟毕": "池壁", "守五": "手", "数官": "树冠", "数长": "树长", "数种": "树种",
    "掌开": "长开", "兼具": "间距", "石拔": "石板", "水井": "水景",
    "挖两壳": "挖了两棵", "挖两棵": "挖了两棵", "五位书": "五位数",
    "指标了": "只标了", "管先": "管线", "交结": "胶粘", "沾死": "粘死",
    "水准": "水景", "稳资": "文字", "笵": "泵", "堆": "队", "王": "往",
    "蕾": "雷", "流": "留", "鞋": "写", "巳": "已",
}

# 重点词（高亮为黄色60号加粗）
KEY_TERMS = [
    "8万", "八万", "十一万", "五位数", "三个雷", "好几万",
    "汀步路", "绿化带", "循环泵", "冠幅", "树冠",
    "石板", "管线", "池壁", "施工队", "业主",
    "动线", "水景", "检修口", "红枫", "元宝枫",
]

# 数字模式（用于高亮）
NUM_PATTERN = re.compile(r'[零一二三四五六七八九十百千万亿\d]+[万千百]*[万千百]*')


# ============================================================
# 工具函数
# ============================================================
def fix(text):
    """修正 Whisper 错字。"""
    for w, c in sorted(CORRECTIONS.items(), key=lambda x: -len(x[0])):
        text = text.replace(w, c)
    return text


def get_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def ts(sec):
    """秒 -> ASS 时间格式 H:MM:SS.cc"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ts_srt(sec):
    """秒 -> SRT 时间格式"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ============================================================
# 分析模块：检测沉默、语气词、重复
# ============================================================
def find_first_word(transcript_path):
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    for seg in data["segments"]:
        for w in seg.get("words", []):
            return w["start"]
    return 5.16


def find_silence(transcript_path, after=0, min_gap=SILENCE_THRESHOLD):
    """检测超过阈值(默认1s)的沉默段落，包括段间。"""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    cuts = []

    # 段内
    for seg in data["segments"]:
        words = seg.get("words", [])
        for i in range(1, len(words)):
            gap = words[i]["start"] - words[i-1]["end"]
            if gap > min_gap:
                s = words[i-1]["end"] + 0.04
                e = words[i]["start"] - 0.04
                if e > s and s >= after:
                    cuts.append((s, e, f"沉默({gap:.1f}s)"))

    # 段间
    for i in range(1, len(data["segments"])):
        prev_w = data["segments"][i-1].get("words", [])
        curr_w = data["segments"][i].get("words", [])
        if prev_w and curr_w:
            gap = curr_w[0]["start"] - prev_w[-1]["end"]
            if gap > min_gap:
                s = prev_w[-1]["end"] + 0.04
                e = curr_w[0]["start"] - 0.04
                if e > s and s >= after:
                    cuts.append((s, e, f"段间沉默({gap:.1f}s)"))

    return cuts


def find_fillers(transcript_path, after=0):
    """检测语气词(嗯啊呃)和废话。"""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    cuts = []
    filler_set = set(FILLER_WORDS.keys())
    longer_fillers = sorted([k for k in FILLER_WORDS if len(k) > 1], key=lambda x: -len(x))

    for seg in data["segments"]:
        words = seg.get("words", [])
        for w in words:
            word = w["word"].strip().lower()
            if not word:
                continue
            # 检测多字语气词优先
            hit = None
            for f in longer_fillers:
                if f in word or word in f:
                    hit = f
                    break
            if hit is None and word in filler_set:
                hit = word

            if hit:
                margin = FILLER_WORDS.get(hit, 0.3)
                s = max(w["start"] - margin, 0)
                e = min(w["end"] + margin, seg["end"])
                if s >= after and e > s:
                    cuts.append((s, e, f"语气词:{hit}"))

    return cuts


def find_repeats(transcript_path, after=0):
    """检测重复内容（段间+段内）。"""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    repeats = []

    # 段间重复
    for i in range(1, len(data["segments"])):
        prev_text = fix(data["segments"][i-1]["text"])
        curr_text = fix(data["segments"][i]["text"])
        prev_w = data["segments"][i-1].get("words", [])
        curr_w = data["segments"][i].get("words", [])

        if not prev_w or not curr_w:
            continue

        # 寻找前段尾部与后段头部的重复
        for n in range(min(40, len(prev_text), len(curr_text)), 3, -1):
            tail = prev_text[-n:].strip("，。！？、； ")
            head = curr_text[:n].strip("，。！？、； ")
            if tail and len(tail) >= 5 and tail[:max(1, len(tail)//2)] in head:
                # 在 prev_words 中定位重复开始位置
                overlap = tail
                start_chars = len(prev_text) - len(overlap)
                char_ratio = start_chars / max(len(prev_text), 1)
                word_idx = int(char_ratio * len(prev_w))
                word_idx = max(0, min(word_idx, len(prev_w) - 1))
                s = prev_w[word_idx]["start"]
                e = prev_w[-1]["end"]
                if s >= after and e > s + 0.3:
                    repeats.append((s, e, f"重复:{overlap[:15]}..."))
                break

    # 段内重复（相邻短语重复）
    for seg in data["segments"]:
        text = fix(seg["text"])
        words = seg.get("words", [])
        if not words:
            continue

        # 按逗号拆分短语
        phrases = re.split(r'(?<=[，,])', text)
        phrases = [p.strip() for p in phrases if p.strip()]

        seen = {}
        for idx, phrase in enumerate(phrases):
            key = phrase.strip("，。！？、；： ")
            if len(key) < 4:
                continue
            if key in seen and idx - seen[key] <= 2:
                # 找到重复
                cum_chars = sum(len(p) for p in phrases[:idx])
                char_ratio = cum_chars / max(len(text), 1)
                word_pos = int(char_ratio * len(words))
                word_pos = max(0, min(word_pos, len(words) - 1))
                end_ratio = (cum_chars + len(phrase)) / max(len(text), 1)
                end_pos = int(end_ratio * len(words))
                end_pos = max(0, min(end_pos, len(words) - 1))
                s = words[word_pos]["start"]
                e = words[end_pos]["end"]
                if s >= after and e > s + 0.3:
                    repeats.append((s, e, f"重复:{key[:15]}..."))
            seen[key] = idx

    # 3. 段内连续短语重复检测（放开阈值，捕获"觉得有用吗,收藏这期,觉得有用吗,收藏这期"模式）
    for seg in data["segments"]:
        text = fix(seg["text"])
        words = seg.get("words", [])
        if not words:
            continue

        # 按逗号/句号拆分
        phrases = re.split(r'(?<=[，,。.!?！？])', text)
        phrases = [p.strip() for p in phrases if p.strip()]

        seen2 = {}
        for idx, phrase in enumerate(phrases):
            key = phrase.strip("，。！？、；： ")
            if len(key) < 3:
                continue
            if key in seen2 and idx - seen2[key] <= 5:
                # 找到重复段 — 标记重复部分
                first_cum = sum(len(phrases[j]) for j in range(seen2[key]))
                end_cum = first_cum + len(phrase) * 2  # approximate
                char_ratio = first_cum / max(len(text), 1)
                word_pos = int(char_ratio * len(words))
                word_pos = max(0, min(word_pos, len(words) - 1))
                end_ratio = min(1.0, end_cum / max(len(text), 1))
                end_pos = int(end_ratio * len(words))
                end_pos = max(0, min(end_pos, len(words) - 1))
                s = words[word_pos]["start"]
                e = words[end_pos]["end"]
                if s >= after and e > s + 0.3:
                    repeats.append((s, e, f"重复:{key[:12]}..."))
            seen2[key] = idx

    return repeats


# ============================================================
# 模糊模块：背景模糊 + 人物锐化
# ============================================================
def blur_video(input_path, output_path, trim_seconds):
    """背景模糊（双重高斯）+ 人物区域锐化。"""
    print(f"  [模糊] 处理中 (trim={trim_seconds:.1f}s)...")
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(3))
    height = int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip = int(trim_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    remaining = total - skip

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    smooth_cx, smooth_cy = None, None
    smooth_rx, smooth_ry = None, None
    alpha_smooth = 0.3
    interval = max(1, remaining // 10)
    processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50, 50))
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cx, cy = fx + fw // 2, fy + fh // 2
            rx, ry = int(fw * 2.5), int(fh * 3.5)

            if smooth_cx is None:
                smooth_cx, smooth_cy = cx, cy
                smooth_rx, smooth_ry = rx, ry
            else:
                a = alpha_smooth
                smooth_cx = int(smooth_cx * (1 - a) + cx * a)
                smooth_cy = int(smooth_cy * (1 - a) + cy * a)
                smooth_rx = int(smooth_rx * (1 - a) + rx * a)
                smooth_ry = int(smooth_ry * (1 - a) + ry * a)

            cv2.ellipse(mask, (smooth_cx, smooth_cy),
                       (smooth_rx, smooth_ry), 0, 0, 360, 255, -1)
        elif smooth_cx is not None:
            cv2.ellipse(mask, (smooth_cx, smooth_cy),
                       (smooth_rx, smooth_ry), 0, 0, 360, 255, -1)

        # 双重背景模糊（更强）
        blurred = cv2.GaussianBlur(frame, (99, 99), 0)
        blurred = cv2.GaussianBlur(blurred, (99, 99), 0)

        # 人物遮罩（轻微羽化保持边缘锐利）
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0) / 255.0
        m3 = np.stack([mask_f] * 3, axis=-1)

        # 合成
        comp = (frame * m3 + blurred * (1 - m3)).astype(np.uint8)

        # 锐化人物区域
        sharpened = cv2.filter2D(comp, -1, sharpen_kernel)
        comp = (sharpened * m3 + comp * (1 - m3)).astype(np.uint8)

        out.write(comp)
        processed += 1
        if processed % interval == 0:
            el = time.time() - t0
            speed = processed / el
            eta = (remaining - processed) / speed
            print(f"    {processed/remaining*100:.0f}% ({speed:.0f}fps, ETA {eta:.0f}s)")

    cap.release()
    out.release()
    print(f"  [模糊] 完成 ({time.time()-t0:.0f}s)")


# ============================================================
# 裁剪模块
# ============================================================
def segment_cut(video_path, audio_path, segments, out_video, out_audio):
    """裁剪视频为保留段并拼接。"""
    n = len(segments)
    if n == 0:
        subprocess.run(["ffmpeg", "-i", str(video_path), "-c", "copy",
                       "-y", str(out_video)], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-i", str(audio_path), "-c", "copy",
                       "-y", str(out_audio)], check=True, capture_output=True)
        return

    seg_dir = TEMP_DIR / "segments"
    seg_dir.mkdir(exist_ok=True)

    concat_v = TEMP_DIR / "concat_v.txt"
    concat_a = TEMP_DIR / "concat_a.txt"
    v_lines, a_lines = [], []

    for i, (s, e) in enumerate(segments):
        d = round(e - s, 3)
        v_seg = seg_dir / f"v{i}.mp4"
        subprocess.run([
            "ffmpeg", "-ss", str(s), "-i", str(video_path),
            "-t", str(d), "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an", "-y", str(v_seg)
        ], check=True, capture_output=True)

        a_wav = seg_dir / f"a{i}.wav"
        subprocess.run([
            "ffmpeg", "-ss", str(s), "-i", str(audio_path),
            "-t", str(d), "-c:a", "pcm_s16le",
            "-y", str(a_wav)
        ], check=True, capture_output=True)

        v_lines.append(f"file '{v_seg}'")
        a_lines.append(f"file '{a_wav}'")

    concat_v.write_text("\n".join(v_lines), encoding="utf-8")
    concat_a.write_text("\n".join(a_lines), encoding="utf-8")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_v),
        "-c", "copy", "-y", str(out_video)
    ], check=True, capture_output=True)

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_a),
        "-c:a", "aac", "-b:a", "128k", "-y", str(out_audio)
    ], check=True, capture_output=True)


def cut_main(blurred_path, audio_path, all_cuts, trim_start):
    """执行裁剪：根据裁剪列表切出保留段。"""
    adj_cuts = [(s - trim_start, e - trim_start) for s, e, _ in all_cuts]
    segments = []
    prev = 0.0
    for s, e in adj_cuts:
        if s > prev + 0.05:
            segments.append((prev, s))
        prev = e
    bdur = get_dur(blurred_path)
    if bdur > prev + 0.05:
        segments.append((prev, bdur))

    print(f"  [裁剪] 保留 {len(segments)} 段, 裁剪 {len(all_cuts)} 处")
    tv = TEMP_DIR / "tv.mp4"
    ta = TEMP_DIR / "ta.aac"
    segment_cut(blurred_path, audio_path, segments, tv, ta)
    return tv, ta


# ============================================================
# 字幕模块：ASS格式，风尚黑体，混合样式，智能换行
# ============================================================
def build_subtitles(transcript_path, cuts, trim_start):
    """构建字幕条目，基于 Whisper 词级时间戳。"""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)

    all_words = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            t = w["word"].strip()
            if t and t not in (",", "，", "。", "?", "？", "！", "!", "、",
                               "；", ";", "：", ":"):
                all_words.append((w["start"], w["end"], t))

    # 分组
    subs = []
    group = []
    punct = set("。！？.!?；;")

    for w_start, w_end, text in all_words:
        group.append((w_start, w_end, text))
        should_break = False
        if text in punct or any(text.endswith(p) for p in punct):
            should_break = True
        if len(group) > 1 and (group[-1][0] - group[-2][1]) > 0.3:
            should_break = True
        if len(group) >= 10:
            should_break = True

        if should_break:
            combined = "".join(w[2] for w in group).strip("，,。.！!？?、；;：:")
            combined = fix(combined)
            if combined:
                subs.append((group[0][0], group[-1][1], combined))
            group = []

    if group:
        combined = "".join(w[2] for w in group).strip("，,。.！!？?、；;：:")
        combined = fix(combined)
        if combined:
            subs.append((group[0][0], group[-1][1], combined))

    # 调整时间轴（考虑裁剪）
    def cut_before(t):
        c = 0.0
        for s, e, _ in cuts:
            if e <= t:
                c += (e - s)
            elif s < t < e:
                c += (t - s)
        return c + trim_start

    adjusted = []
    for s, e, text in subs:
        ns = s - cut_before(s)
        ne = e - cut_before(e)
        if ne > ns + 0.15 and text:
            adjusted.append((ns, ne, text))

    # 合并 ≤2字条目到前一句
    merged = []
    for s, e, text in adjusted:
        if len(text) <= 2 and merged:
            prev_s, prev_e, prev_text = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e), prev_text + text)
        elif len(text) > 2:
            merged.append((s, e, text))

    return merged


def format_subtitle_line(text):
    """
    智能换行规则：
    - 逗号/句号后如果只剩1-2个字符 → 换行到下一行
    - 不允许 "一句话说完，又" 同行的格式
    """
    for punct in "，,。.!?？；;、":
        # 从右往左找标点
        idx = text.rfind(punct)
        if idx >= 0 and idx < len(text) - 1:
            after = text[idx+1:].strip()
            if len(after) <= 2:
                text = text[:idx+1] + "\\N" + after
                break
    return text


def auto_wrap_text(text, max_chars=22):
    """
    自动折行：每行最多 max_chars 字符，在标点处断开。
    """
    if len(text) <= max_chars:
        return text

    result = ""
    while len(text) > max_chars:
        # 在 max_chars 范围内找最后一个标点断行
        break_pos = -1
        for i in range(max_chars, max_chars // 2, -1):
            if i < len(text) and text[i] in "，,。.!?？；;、：: ":
                break_pos = i + 1
                break
        if break_pos == -1:
            break_pos = max_chars

        result += text[:break_pos].strip() + "\\N"
        text = text[break_pos:].strip()

    result += text
    return result


def is_keyword(word):
    """判断是否为需要高亮的重点词。"""
    if word in KEY_TERMS:
        return True
    if NUM_PATTERN.fullmatch(word):
        return True
    # 包含数字的词
    if any(ch.isdigit() or ch in "零一二三四五六七八九十百千万亿" for ch in word):
        return True
    return False


def build_ass_text(text):
    """
    将普通文本转为带 ASS 格式标记的文本。
    重点词加大黄色加粗，普通词白色。
    """
    if not text:
        return text

    # 使用配置的字体大小
    tags_close = f"{{\\fs{FONT_SIZE}\\c&H00FFFFFF&\\b0}}"
    tags_hl = f"{{\\fs{FONT_SIZE_HIGHLIGHT}\\c&H0000FFFF&\\b1}}"

    result = ""
    i = 0
    while i < len(text):
        matched = False
        # 先匹配 KEY_TERMS（长词优先）
        for term in sorted(KEY_TERMS, key=lambda x: -len(x)):
            if text[i:i+len(term)] == term:
                result += tags_hl + term + tags_close
                i += len(term)
                matched = True
                break
        if matched:
            continue
        # 再匹配数字模式
        m = NUM_PATTERN.match(text, i)
        if m and len(m.group()) >= 1:
            word = m.group()
            result += tags_hl + word + tags_close
            i += len(word)
            continue
        # 普通字符
        result += text[i]
        i += 1

    return result


def write_ass(subtitles, path, width, height):
    """生成 ASS 字幕文件（支持混合样式、智能换行）。"""
    lines = [
        "[Script Info]",
        f"Title: 口播字幕",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{FONT_NAME},{FONT_SIZE},{FONT_COLOR},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,1,2,10,10,120,134",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for s, e, text in subtitles:
        # 按原始标点换行
        display_text = format_subtitle_line(text)
        # 手动再加一层：超长文本自动在标点处折行
        # 每行最多25个字
        display_text = auto_wrap_text(display_text, max_chars=22)
        # ASS 标记（重点词高亮）
        marked_text = build_ass_text(display_text)
        if marked_text == display_text:
            marked_text = display_text

        lines.append(
            f"Dialogue: 0,{ts(s)},{ts(e)},Default,,0,0,0,,{marked_text}"
        )

    path.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"  [字幕] ASS: {len(subtitles)} 条 ({FONT_NAME} {FONT_SIZE}号)")


# ============================================================
# 渲染模块：合成最终视频
# ============================================================
def mux_av(video_path, audio_path, output_path):
    """合成视频+音频。"""
    subprocess.run([
        "ffmpeg", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-y", str(output_path)
    ], check=True, capture_output=True)


def burn_ass(video_path, ass_path, output_path):
    """烧录 ASS 字幕到视频。"""
    print(f"  [渲染] 烧录字幕...")
    ass_esc = str(ass_path).replace("\\", "/").replace(":", "\\:")
    subprocess.run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"ass='{ass_esc}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "copy",
        "-y", str(output_path)
    ], check=True, capture_output=True)


# ============================================================
# 主流程
# ============================================================
def main():
    print()
    print("=" * 60)
    print("  口播视频剪辑 Agent v1.1")
    print("  Sub-Agent for 口播 editing")
    print("=" * 60)

    # ========================================
    # 第零步：清理
    # ========================================
    for p in [TEMP_DIR / "segments", TEMP_DIR / "tv.mp4",
              TEMP_DIR / "ta.aac", TEMP_DIR / "muxed.mp4",
              OUTPUT_DIR / "final.mp4"]:
        if p.exists():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()

    # ========================================
    # 第一步：分析
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 1: 分析")
    print("─" * 50)

    trim_start = find_first_word(TRANSCRIPT_FILE)
    print(f"  📍 片头空档: 0-{trim_start:.1f}s")

    # 找沉默
    silence_cuts = find_silence(TRANSCRIPT_FILE, after=trim_start)
    total_silence = sum(e - s for s, e, _ in silence_cuts)
    print(f"  🔇 沉默(>{SILENCE_THRESHOLD}s): {len(silence_cuts)} 处, 共 {total_silence:.1f}s")

    # 找语气词
    filler_cuts = find_fillers(TRANSCRIPT_FILE, after=trim_start)
    total_filler = sum(e - s for s, e, _ in filler_cuts)
    print(f"  🗣️ 语气词: {len(filler_cuts)} 处, 共 {total_filler:.1f}s")
    if filler_cuts:
        for s, e, reason in filler_cuts[:10]:
            print(f"    {s:.1f}s-{e:.1f}s  {reason}")

    # 找重复
    repeat_cuts = find_repeats(TRANSCRIPT_FILE, after=trim_start)
    total_repeat = sum(e - s for s, e, _ in repeat_cuts)
    print(f"  🔄 重复: {len(repeat_cuts)} 处, 共 {total_repeat:.1f}s")

    # 合并所有裁剪段
    all_cuts_raw = silence_cuts + filler_cuts + repeat_cuts
    # 合并重叠段
    all_cuts = []
    for s, e, reason in sorted(all_cuts_raw, key=lambda x: x[0]):
        if not all_cuts:
            all_cuts.append((s, e, reason))
        else:
            last_s, last_e, last_reason = all_cuts[-1]
            if s <= last_e + 0.1:
                # 重叠或相邻，合并
                all_cuts[-1] = (last_s, max(last_e, e), f"{last_reason} + {reason}")
            else:
                all_cuts.append((s, e, reason))

    total_cut = sum(e - s for s, e, _ in all_cuts)
    print(f"  ✂️ 总计: {len(all_cuts)} 处裁剪段, 共 {total_cut:.1f}s")

    if all_cuts:
        print("  ── 裁剪列表 ──")
        for s, e, reason in all_cuts:
            dur = e - s
            print(f"    {s:.1f}s-{e:.1f}s ({dur:.1f}s)  {reason}")

    # ========================================
    # 第二步：提取音频
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 2: 提取音频")
    print("─" * 50)

    raw_audio = TEMP_DIR / "raw.aac"
    subprocess.run([
        "ffmpeg", "-i", str(ORIGINAL_VIDEO), "-vn",
        "-c:a", "copy", "-y", str(raw_audio)
    ], check=True, capture_output=True)
    audio_t = TEMP_DIR / "audio_t.aac"
    subprocess.run([
        "ffmpeg", "-i", str(raw_audio), "-ss", str(trim_start),
        "-c", "copy", "-y", str(audio_t)
    ], check=True, capture_output=True)
    print(f"  OK ({get_dur(audio_t):.1f}s)")

    # ========================================
    # 第三步：背景模糊
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 3: 背景模糊 + 人物锐化")
    print("─" * 50)

    blurred = TEMP_DIR / "blurred.mp4"
    blur_video(ORIGINAL_VIDEO, blurred, trim_start)

    # ========================================
    # 第四步：裁剪
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 4: 裁剪沉默/语气词/重复")
    print("─" * 50)

    tv, ta = cut_main(blurred, audio_t, all_cuts, trim_start)

    # ========================================
    # 第五步：字幕
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 5: 生成字幕 (ASS)")
    print("─" * 50)

    # 获取视频宽高
    cap = cv2.VideoCapture(str(tv))
    vw = int(cap.get(3))
    vh = int(cap.get(4))
    cap.release()

    subs = build_subtitles(TRANSCRIPT_FILE, all_cuts, trim_start)
    ass_path = TEMP_DIR / "subtitles.ass"
    write_ass(subs, ass_path, vw, vh)

    # ========================================
    # 第六步：合成
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 6: 合成视频")
    print("─" * 50)

    muxed = TEMP_DIR / "muxed.mp4"
    mux_av(tv, ta, muxed)

    # ========================================
    # 第七步：烧录字幕
    # ========================================
    print("\n" + "─" * 50)
    print("  Step 7: 烧录字幕")
    print("─" * 50)

    final = OUTPUT_DIR / "final.mp4"
    burn_ass(muxed, ass_path, final)

    # ========================================
    # 完成
    # ========================================
    od = get_dur(ORIGINAL_VIDEO)
    fd = get_dur(final)
    print()
    print("=" * 60)
    print("  ✅ 完成!")
    print(f"  ⏱  {od:.1f}s → {fd:.1f}s (剪 {od-fd:.1f}s)")
    print(f"  📝 字幕: {len(subs)} 条 (ASS格式)")
    print(f"  🎨 字体: {FONT_NAME} {FONT_SIZE}号 / 重点词 {FONT_SIZE_HIGHLIGHT}号黄色")
    print(f"  🔇 沉默: {len(silence_cuts)} 处 | 🗣️ 语气词: {len(filler_cuts)} 处 | 🔄 重复: {len(repeat_cuts)} 处")
    print(f"  📂 输出: {final}")
    print("=" * 60)


if __name__ == "__main__":
    main()
