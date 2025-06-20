import os
import subprocess
import time
from pathlib import Path

import requests
from faster_whisper import WhisperModel
from core.config import config


def extract_audio(video_path: str, audio_path: str) -> None:
    """Извлекает звуковую дорожку в 16kHz моно WAV с помощью ffmpeg."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le" "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
        "-y",
    ]
    subprocess.run(command, check=True)


def transcribe_audio(audio_path: str, language: str = "ru") -> str:
    """Транскрибирует аудио в текст с помощью Whisper."""
    asr = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(audio_path, language=language)
    print(segments)
    return " ".join([segment.text for segment in segments])


def summarize_text(text: str, model: str = "deepseek-r1:8b") -> str:
    """Отправляет текст в локальную модель Ollama и возвращает пересказ."""
    prompt = (
        "Сделай подробный, но максимально сжатый пересказ этого текста на русском языке. "
        "Сохрани все ключевые детали и контекст, избегай потери важных смыслов. "
        "Не добавляй размышлений, комментариев или формат <think>. "
        "Выдай связный, логичный и компактный пересказ, который можно использовать вместо оригинала:"
        f"{text}\n\n"
    )
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    print(resp.json())
    return resp.json()["message"]["content"]


def summarize_audio(audio_path: str, language: str = "ru") -> str:
    """
    Транскрибирует аудиофайл и создает выжимку из текста.
    Args:
        audio_path: Путь к аудиофайлу.
        language: Язык аудио.
    Returns:
        Выжимка из аудио.
    """
    print("🗣️  Транскрибирую…")
    transcript = transcribe_audio(audio_path, language=language)
    print("\n🔊 Распознанный текст (обрезано):")
    print(transcript[:300] + ("…" if len(transcript) > 300 else ""))

    print("\n📝 Запрашиваю выжимку у Ollama…")
    summary = summarize_text(transcript, model=config.OLLAMA_MODEL)
    return summary


def summarize_video(video_path: str, language: str = "ru") -> str:
    """
    Создает выжимку из видеофайла.
    Args:
        video_path: Путь к видеофайлу.
        language: Язык аудио в видео.
    Returns:
        Выжимка из видео.
    """
    start_time = time.time()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Файл {video_path} не найден")

    with Path(config.TMP_AUDIO) as tmp_audio_path:
        try:
            print("🎬 Извлекаю аудио…")
            extract_audio(video_path, str(tmp_audio_path))

            summary = summarize_audio(str(tmp_audio_path), language=language)

            print("\n📌 Итоговый пересказ:")
            print(summary)

            elapsed = time.time() - start_time
            print(f"\n⏱️ Время выполнения: {elapsed:.1f} секунд")
            return summary
        finally:
            if tmp_audio_path.exists():
                os.remove(tmp_audio_path)
