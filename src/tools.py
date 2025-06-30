import os
import subprocess
import time
from pathlib import Path

import requests
import re
from faster_whisper import WhisperModel
from core.config import config
from qdrant_manager import QdrantManager


def extract_audio(video_path: str, audio_path: str) -> None:
    """Извлекает звуковую дорожку в 16kHz моно WAV с помощью ffmpeg."""
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
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
    response = resp.json()["message"]["content"]
    cleaned_content = re.sub(r"<think>.*?</think>\n?", "", response, flags=re.DOTALL)
    print(cleaned_content)
    return cleaned_content


def summarize_audio(audio_path: str, language: str = "ru", save_to_qdrant: bool = True) -> str:
    """
    Транскрибирует аудиофайл и создает выжимку из текста.
    Args:
        audio_path: Путь к аудиофайлу.
        language: Язык аудио.
        save_to_qdrant: Сохранять ли выжимку в Qdrant.
    Returns:
        Выжимка из аудио.
    """
    # Проверяем, есть ли уже выжимка для этого файла
    if save_to_qdrant:
        qdrant_manager = QdrantManager()
        existing = qdrant_manager.check_file_exists(audio_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {Path(audio_path).name}")
            print(f"📅 Создана: {existing['created_at']}")
            return existing['summary']
    
    print("🗣️  Транскрибирую…")
    transcript = transcribe_audio(audio_path, language=language)
    print("\n🔊 Распознанный текст (обрезано):")
    print(transcript)

    print("\n📝 Запрашиваю выжимку у Ollama…")
    summary = summarize_text(transcript, model=config.OLLAMA_MODEL)
    
    # Сохраняем в Qdrant
    if save_to_qdrant:
        try:
            metadata = {
                "language": language,
                "transcript_length": len(transcript),
                "ollama_model": config.OLLAMA_MODEL
            }
            qdrant_manager.save_summary(audio_path, summary, "audio", metadata)
        except Exception as e:
            print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
    
    return summary


def summarize_video(video_path: str, language: str = "ru", save_to_qdrant: bool = True) -> str:
    """
    Создает выжимку из видеофайла.
    Args:
        video_path: Путь к видеофайлу.
        language: Язык аудио в видео.
        save_to_qdrant: Сохранять ли выжимку в Qdrant.
    Returns:
        Выжимка из видео.
    """
    start_time = time.time()
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Файл {video_path} не найден")

    # Проверяем, есть ли уже выжимка для этого файла
    if save_to_qdrant:
        qdrant_manager = QdrantManager()
        existing = qdrant_manager.check_file_exists(video_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {Path(video_path).name}")
            print(f"📅 Создана: {existing['created_at']}")
            return existing['summary']

    with Path(config.TMP_AUDIO) as tmp_audio_path:
        try:
            print("🎬 Извлекаю аудио…")
            extract_audio(video_path, str(tmp_audio_path))

            # Не сохраняем промежуточную аудио-выжимку в Qdrant, только финальную видео-выжимку
            summary = summarize_audio(str(tmp_audio_path), language=language, save_to_qdrant=False)

            print("\n📌 Итоговый пересказ:")
            print(summary)

            elapsed = time.time() - start_time
            print(f"\n⏱️ Время выполнения: {elapsed:.1f} секунд")
            
            # Сохраняем видео-выжимку в Qdrant
            if save_to_qdrant:
                try:
                    metadata = {
                        "language": language,
                        "processing_time": elapsed,
                        "ollama_model": config.OLLAMA_MODEL
                    }
                    qdrant_manager.save_summary(video_path, summary, "video", metadata)
                except Exception as e:
                    print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
            
            return summary
        finally:
            if tmp_audio_path.exists():
                os.remove(tmp_audio_path)


def search_summaries(query: str, limit: int = 5, min_score: float = 0.3) -> str:
    """
    Ищет похожие выжимки в Qdrant по запросу.
    Args:
        query: Поисковый запрос.
        limit: Максимальное количество результатов.
        min_score: Минимальный порог схожести (0.0-1.0).
    Returns:
        Форматированный список найденных выжимок.
    """
    try:
        qdrant_manager = QdrantManager()
        results = qdrant_manager.search_similar_summaries(query, limit=limit, min_score=min_score)
        
        if not results:
            return f"🤷 Похожие выжимки не найдены (мин. схожесть: {min_score:.1f})."
        
        output = f"🔍 Найдено {len(results)} похожих выжимок:\n\n"
        
        for i, result in enumerate(results, 1):
            score = result['score']
            file_name = result['file_name']
            file_type = result['file_type']
            created_at = result['created_at'][:10]  # Только дата
            summary_preview = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
            chunks_count = result.get('chunks_count', 1)
            is_chunked = result.get('is_chunked', False)
            
            output += f"{i}. 📁 {file_name} ({file_type})\n"
            output += f"   📊 Схожесть: {score:.3f}\n"
            output += f"   📅 Создана: {created_at}\n"
            
            if is_chunked:
                output += f"   🧩 Чанков: {chunks_count}\n"
            
            output += f"   📝 Выжимка: {summary_preview}\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Ошибка при поиске: {e}"


def summarize_text_and_save(text: str, file_path: str = "manual_input", model: str = None) -> str:
    """
    Создает выжимку из текста и сохраняет в Qdrant.
    Args:
        text: Исходный текст.
        file_path: Путь к файлу (или описание источника).
        model: Модель для создания выжимки.
    Returns:
        Выжимка из текста.
    """
    if model is None:
        model = config.OLLAMA_MODEL
    
    summary = summarize_text(text, model)
    
    # Сохраняем в Qdrant
    try:
        qdrant_manager = QdrantManager()
        metadata = {
            "text_length": len(text),
            "ollama_model": model,
            "source": "manual_input"
        }
        qdrant_manager.save_summary(file_path, summary, "text", metadata)
    except Exception as e:
        print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
    
    return summary
