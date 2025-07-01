import subprocess
import time

from faster_whisper import WhisperModel


def extract_audio(video_path: str, audio_path: str, speed_multiplier: float = 1.0) -> None:
    """
    Извлекает звуковую дорожку в 16kHz моно WAV с помощью ffmpeg.
    
    Args:
        video_path: Путь к видеофайлу
        audio_path: Путь для сохранения аудио
        speed_multiplier: Множитель скорости (2.0 = 2x быстрее, 0.5 = 2x медленнее)
    """
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
    
    command.extend([audio_path, "-y"])
    subprocess.run(command, check=True)


def speed_up_audio(input_path: str, output_path: str, speed_multiplier: float = 2.0) -> None:
    """
    Ускоряет существующий аудиофайл.
    
    Args:
        input_path: Путь к исходному аудиофайлу
        output_path: Путь для сохранения ускоренного аудио
        speed_multiplier: Множитель скорости (2.0 = 2x быстрее)
    """
    print(f"⚡ Ускоряю аудио в {speed_multiplier}x раз...")
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
        output_path,
        "-y"
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
