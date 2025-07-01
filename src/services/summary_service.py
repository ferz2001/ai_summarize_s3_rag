from pathlib import Path

from services.local_models_service import LocalModelsService
from utils.media_processing import extract_audio, speed_up_audio
from core.config import config


class SummaryService:
    """Сервис суммаризации - только создание саммаризаций без сохранения в RAG."""
    
    def __init__(self):
        self.local_models = LocalModelsService() if config.USE_LOCAL_MODELS else None
    
    async def summarize_text(
        self, 
        text: str, 
        file_path: str = "manual_input.txt"
    ) -> dict:
        """Создаёт суммаризацию текста."""
        # Генерируем суммаризацию
        if config.USE_LOCAL_MODELS:
            summary = await self.local_models.chat_completion([
                {
                    "role": "system",
                    "content": "Ты помощник для создания выжимок. Создавай краткие, но полные пересказы на русском языке."
                },
                {
                    "role": "user",
                    "content": f"Создай краткую выжимку следующего текста:\n\n{text}"
                }
            ])
        else:
            # Используем OpenAI через LangChain
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.7
            )
            prompt = f"Создай краткую выжимку следующего текста:\n\n{text}"
            response = await llm.ainvoke(prompt)
            summary = response.content
        
        return {
            "summary": summary,
            "file_path": file_path,
            "original_text": text,
            "original_length": len(text),
            "summary_length": len(summary)
        }
    
    async def summarize_audio(
        self,
        file_path: str,
        speed_multiplier: float = 2.0
    ) -> dict:
        """Суммаризирует аудио файл."""
        # Ускоряем аудио если нужно
        audio_path = file_path
        if speed_multiplier != 1.0:
            audio_path = await speed_up_audio(file_path, speed_multiplier)
        
        # Транскрибируем
        if config.USE_LOCAL_MODELS:
            # Пока используем OpenAI для транскрипции, даже если локальные модели включены
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            with open(audio_path, "rb") as audio_file:
                transcription_response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcription = transcription_response.text
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            with open(audio_path, "rb") as audio_file:
                transcription_response = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcription = transcription_response.text
        
        # Очищаем временный файл
        if speed_multiplier != 1.0:
            Path(audio_path).unlink(missing_ok=True)
        
        # Суммаризируем транскрипцию
        result = await self.summarize_text(transcription, file_path)
        result["transcription"] = transcription
        result["file_type"] = "audio"
        return result
    
    async def summarize_video(
        self,
        file_path: str,
        speed_multiplier: float = 2.0
    ) -> dict:
        """Суммаризирует видео файл."""
        # Извлекаем аудио
        audio_path = await extract_audio(file_path, speed_multiplier)
        
        try:
            # Транскрибируем аудио
            if config.USE_LOCAL_MODELS:
                # Пока используем OpenAI для транскрипции, даже если локальные модели включены
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
                with open(audio_path, "rb") as audio_file:
                    transcription_response = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    transcription = transcription_response.text
            else:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
                with open(audio_path, "rb") as audio_file:
                    transcription_response = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    transcription = transcription_response.text
            
            # Суммаризируем
            result = await self.summarize_text(transcription, file_path)
            result["transcription"] = transcription
            result["file_type"] = "video"
            return result
        
        finally:
            # Очищаем временный аудио файл
            Path(audio_path).unlink(missing_ok=True)


# Глобальный экземпляр
summary_service = SummaryService() 