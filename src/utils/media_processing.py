import asyncio
import tempfile
import time
from faster_whisper import WhisperModel


async def extract_audio(video_path: str, speed_multiplier: float = 1.0) -> str:
    """
    Извлекает звуковую дорожку в 16kHz моно WAV с помощью ffmpeg.
    
    Args:
        video_path: Путь к видеофайлу
        speed_multiplier: Множитель скорости (2.0 = 2x быстрее, 0.5 = 2x медленнее)
        
    Returns:
        Путь к извлечённому аудио файлу
    """
    # Создаём временный файл для аудио
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
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
        "1"
    ]
    
    # Добавляем ускорение, если нужно
    if speed_multiplier != 1.0:
        print(f"⚡ Ускоряю аудио в {speed_multiplier}x раз для быстрой обработки")
        command.extend(["-filter:a", f"atempo={speed_multiplier}"])
    
    command.extend([temp_audio, "-y"])
    
    # Выполняем асинхронно
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if process.returncode != 0:
        raise Exception("Ошибка извлечения аудио из видео")
    
    return temp_audio


async def speed_up_audio(input_path: str, speed_multiplier: float = 2.0) -> str:
    """
    Ускоряет существующий аудиофайл.
    
    Args:
        input_path: Путь к исходному аудиофайлу
        speed_multiplier: Множитель скорости (2.0 = 2x быстрее)
        
    Returns:
        Путь к ускоренному аудио файлу
    """
    print(f"⚡ Ускоряю аудио в {speed_multiplier}x раз...")
    
    # Создаём временный файл для результата
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-filter:a",
        f"atempo={speed_multiplier}",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_output,
        "-y"
    ]
    
    # Выполняем асинхронно
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if process.returncode != 0:
        raise Exception("Ошибка ускорения аудио")
    
    return temp_output


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
