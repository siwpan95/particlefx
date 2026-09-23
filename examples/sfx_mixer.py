"""
真实音效加载与混合模块 v1.0

用 ffmpeg 解码任意音频格式 → numpy 数组，支持时间轴多层混合。
不需要 pydub，只需要 ffmpeg + numpy。

用法:
    from sfx_mixer import load_audio, mix_timeline, write_wav

    # 加载音效文件（音效库根目录由调用方决定，可用环境变量 SOUNDS_DIR / SFX_ROOT 指定）
    rain = load_audio("<SOUNDS_DIR>/light-rain-01.mp3")
    thunder = load_audio("<SOUNDS_DIR>/thunder7.wav")

    # 按时间轴混合（音效, 起始秒, 音量, 淡入秒, 淡出秒）
    mixed = mix_timeline([
        (rain,    5.0,  0.6, 2.0, 2.0),   # 雨声@5s
        (thunder, 60.0, 1.0, 0.0, 0.0),   # 雷声@60s
    ], duration_sec=150.0)

    write_wav("output.wav", mixed)
"""

import subprocess
import os
import numpy as np


def load_audio(path, target_sr=44100, max_duration=None):
    """用 ffmpeg 解码任意音频文件 → float64 单声道数组

    Args:
        path: mp3/wav/flac 等音频文件路径
        target_sr: 目标采样率 (Hz)
        max_duration: 最大加载时长（秒），None=全部

    Returns:
        np.ndarray (float64, -1.0 ~ 1.0)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")

    duration_args = []
    if max_duration is not None:
        duration_args = ["-t", str(max_duration)]

    # ffmpeg 解码 → 16-bit signed PCM → stdout
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", path,
        *duration_args,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败: {path}\n{result.stderr.decode()[:500]}")

    raw = result.stdout
    if len(raw) < 2:
        # 空文件
        return np.zeros(0, dtype=np.float64)

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
    return data


def get_duration(path):
    """获取音频文件时长（秒）"""
    cmd = [
        "ffmpeg", "-i", path,
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # 从 stderr 解析 Duration 行
    import re
    m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if m:
        h, mi, s, cs = map(int, m.groups())
        return h * 3600 + mi * 60 + s + cs / 100.0
    return 0.0


def _apply_fade(audio, sample_rate, fade_in_sec=0.0, fade_out_sec=0.0):
    """对音频片段应用淡入淡出（原地修改）"""
    n = len(audio)
    if n == 0:
        return audio

    # 淡入
    if fade_in_sec > 0:
        fi_samples = min(int(fade_in_sec * sample_rate), n)
        if fi_samples > 1:
            t = np.linspace(0, 1, fi_samples)
            audio[:fi_samples] *= t

    # 淡出
    if fade_out_sec > 0:
        fo_samples = min(int(fade_out_sec * sample_rate), n)
        if fo_samples > 1:
            t = np.linspace(1, 0, fo_samples)
            audio[-fo_samples:] *= t

    return audio


def _loop_audio(audio, target_length):
    """循环音频到目标长度"""
    if len(audio) == 0:
        return audio
    if len(audio) >= target_length:
        return audio[:target_length]

    repeats = int(np.ceil(target_length / len(audio)))
    looped = np.tile(audio, repeats)
    return looped[:target_length]


def mix_timeline(sounds, duration_sec, sample_rate=44100):
    """将多个音效按时间轴混合

    Args:
        sounds: list of tuple:
            (audio_array, start_sec, volume, fade_in_sec, fade_out_sec)
            或 (audio_array, start_sec, volume, fade_in_sec, fade_out_sec, play_duration_sec)
            play_duration_sec: 可选，播放时长。如果源音频不够长则循环填充。
        duration_sec: 总时长（秒）
        sample_rate: 采样率

    Returns:
        np.ndarray (float64) 混合后的音频
    """
    total_samples = int(duration_sec * sample_rate)
    mix = np.zeros(total_samples, dtype=np.float64)

    for entry in sounds:
        audio = entry[0]
        start_sec, volume, fade_in, fade_out = entry[1:5]
        play_duration = entry[5] if len(entry) > 5 else None

        if len(audio) == 0:
            continue

        start_sample = int(start_sec * sample_rate)
        if start_sample >= total_samples:
            continue

        # 如果指定了播放时长且源音频不够长，循环填充
        seg = audio.copy()
        if play_duration is not None:
            target_len = int(play_duration * sample_rate)
            if len(seg) < target_len:
                seg = _loop_audio(seg, target_len)
            else:
                seg = seg[:target_len]

        # 应用淡入淡出
        if fade_in > 0 or fade_out > 0:
            _apply_fade(seg, sample_rate, fade_in, fade_out)

        seg *= volume

        # 放置到时间轴
        available = total_samples - start_sample
        seg = seg[:available]

        mix[start_sample:start_sample + len(seg)] += seg

    return mix


def normalize_int16(audio, peak=0.85):
    """归一化并转为 int16"""
    max_val = max(np.abs(audio).max(), 1e-9)
    normalized = (audio / max_val * peak * 32767)
    return np.clip(normalized, -32767, 32767).astype(np.int16)


def write_wav(path, audio, sample_rate=44100):
    """写入 WAV 文件（单声道 16-bit）"""
    import wave
    samples = normalize_int16(audio)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
