import os
import subprocess
import time
from pathlib import Path
import tempfile
import shutil
import httpx

from faster_whisper import WhisperModel
from core.config import config
from services.qdrant_service import QdrantService


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
    """
    Транскрибирует аудиофайл с помощью Whisper.
    Args:
        audio_path: Путь к аудиофайлу.
        language: Язык аудио.
    Returns:
        Текст транскрипции.
    """
    print(f"🗣️ Начинаю транскрипцию аудио: {audio_path}")
    start_time = time.time()
    
    # Используем локальную модель Whisper
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(
        audio_path, 
        language=language,
        condition_on_previous_text=False,
        word_timestamps=False
    )
    
    transcript = " ".join([segment.text for segment in segments])
    
    print(f"✅ Транскрипция завершена за {time.time() - start_time:.1f}с")
    print(f"📝 Получено {len(transcript)} символов текста")
    
    return transcript.strip()


async def summarize_text(text: str, model: str = None) -> str:
    """Создает пересказ текста с помощью локальных моделей."""
    prompt = (
        "Сделай подробный, но максимально сжатый пересказ этого текста на русском языке. "
        "Сохрани все ключевые детали и контекст, избегай потери важных смыслов. "
        "Не добавляй размышлений, комментариев или формат <think>. "
        "Выдай связный, логичный и компактный пересказ, который можно использовать вместо оригинала:\n\n"
        f"{text}"
    )
    if config.USE_LOCAL_MODELS:
        # Используем только локальные модели, без fallback
        from services.local_models_service import LocalModelsService
        local_models = LocalModelsService()
        summary = await local_models.chat_completion([
            {
                "role": "system",
                "content": "Ты помощник для создания выжимок. Создавай краткие, но полные пересказы на русском языке."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ], model)
        print(summary)
        return summary
    else:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            print(content)
            return content


async def summarize_audio(audio_path: str, language: str = "ru", save_to_qdrant: bool = True) -> str:
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
        qdrant_service = QdrantService()
        existing = await qdrant_service.check_file_exists(audio_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {Path(audio_path).name}")
            print(f"📅 Создана: {existing['created_at']}")
            return existing['summary']
    
    print("🗣️  Транскрибирую…")
    transcript = transcribe_audio(audio_path, language=language)
    print("\n🔊 Распознанный текст (обрезано):")
    print(transcript)

    print("\n📝 Создаю выжимку с помощью ИИ…")
    summary = await summarize_text(transcript)
    
    # Сохраняем в Qdrant
    if save_to_qdrant:
        try:
            metadata = {
                "language": language,
                "transcript_length": len(transcript),
                "model": config.LOCAL_CHAT_MODEL if config.USE_LOCAL_MODELS else "gpt-4o-mini"
            }
            await qdrant_service.save_summary(audio_path, summary, "audio", metadata)
        except Exception as e:
            print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
    
    return summary


async def summarize_video(video_path: str, language: str = "ru", save_to_qdrant: bool = True) -> str:
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
        qdrant_service = QdrantService()
        existing = await qdrant_service.check_file_exists(video_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {Path(video_path).name}")
            print(f"📅 Создана: {existing['created_at']}")
            return existing['summary']

    with Path(config.TMP_AUDIO) as tmp_audio_path:
        try:
            print("🎬 Извлекаю аудио…")
            extract_audio(video_path, str(tmp_audio_path))

            # Не сохраняем промежуточную аудио-выжимку в Qdrant, только финальную видео-выжимку
            summary = await summarize_audio(str(tmp_audio_path), language=language, save_to_qdrant=False)

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
                        "model": config.LOCAL_CHAT_MODEL if config.USE_LOCAL_MODELS else "gpt-4o-mini"
                    }
                    await qdrant_service.save_summary(video_path, summary, "video", metadata)
                except Exception as e:
                    print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
            
            return summary
        finally:
            if tmp_audio_path.exists():
                os.remove(tmp_audio_path)


async def search_summaries(query: str, limit: int = 5, min_score: float = 0.3) -> str:
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
        qdrant_service = QdrantService()
        results = await qdrant_service.search_similar_summaries(query, limit=limit, min_score=min_score)
        
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


async def summarize_text_and_save(text: str, file_path: str = "manual_input", model: str = None) -> str:
    """
    Создает выжимку из текста и сохраняет в Qdrant.
    Args:
        text: Исходный текст.
        file_path: Путь к файлу (или описание источника).
        model: Модель для создания выжимки.
    Returns:
        Выжимка из текста.
    """
    summary = await summarize_text(text, model)

    try:
        qdrant_service = QdrantService()
        metadata = {
            "text_length": len(text),
            "model": config.LOCAL_CHAT_MODEL if config.USE_LOCAL_MODELS else "gpt-4o-mini",
            "source": "manual_input"
        }
        await qdrant_service.save_summary(file_path, summary, "text", metadata)
    except Exception as e:
        print(f"⚠️  Не удалось сохранить в Qdrant: {e}")
    
    return summary
