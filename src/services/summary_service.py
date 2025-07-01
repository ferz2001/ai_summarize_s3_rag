import os
from pathlib import Path
from typing import List
from datetime import datetime

from core.config import config
from services.qdrant_service import QdrantService
from utils.media_processing import transcribe_audio, extract_audio, summarize_text
from schemas.summary import SummaryResponse
from schemas.search import SearchResult


class SummaryService:
    """Сервис для работы с выжимками."""
    
    def __init__(self):
        self.qdrant_service = QdrantService()
    
    async def _create_summary_with_local_model(self, text: str) -> str:
        """Создает выжимку с помощью локальной модели или OpenAI."""
        return await summarize_text(text)
    
    async def create_text_summary(
        self, 
        text: str, 
        file_path: str = "manual_input"
    ) -> SummaryResponse:
        """Создает выжимку из текста."""
        print(f"📝 Создание выжимки из текста (длина: {len(text)} символов)")
        
        # Создаем выжимку с помощью локальной модели
        summary = await self._create_summary_with_local_model(text)
        
        # Сохраняем в Qdrant
        metadata = {
            "text_length": len(text),
            "model": config.LOCAL_CHAT_MODEL,
            "source": "api_text_input"
        }
        
        point_ids = await self.qdrant_service.save_summary(
            file_path=file_path,
            summary=summary,
            file_type="text",
            metadata=metadata
        )
        
        print(f"✅ Выжимка создана и сохранена (ID: {point_ids[0]})")
        
        return SummaryResponse(
            id=point_ids[0],
            summary=summary,
            file_name=Path(file_path).name,
            file_type="text",
            chunks_count=len(point_ids),
            is_chunked=len(point_ids) > 1,
            created_at=datetime.now().isoformat()
        )
    
    async def create_audio_summary(
        self,
        file_path: str,
        file_name: str,
        language: str = "ru"
    ) -> SummaryResponse:
        """Создает выжимку из аудиофайла."""
        print(f"🎵 Обработка аудиофайла: {file_name}")
        
        # Проверяем существующую выжимку
        existing = await self.qdrant_service.check_file_exists(file_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {file_name}")
            return SummaryResponse(
                id=existing["id"],
                summary=existing["summary"],
                file_name=file_name,
                file_type="audio",
                chunks_count=existing.get("chunks_count", 1),
                is_chunked=existing.get("is_chunked", False),
                created_at=existing["created_at"]
            )
        
        # Транскрибируем аудио
        print("🗣️ Транскрибируем аудио...")
        transcript = transcribe_audio(file_path, language)
        print(f"📝 Транскрипция получена (длина: {len(transcript)} символов)")
        
        # Создаем выжимку
        summary = await self._create_summary_with_local_model(transcript)
        
        # Сохраняем в Qdrant
        metadata = {
            "language": language,
            "transcript_length": len(transcript),
            "model": config.LOCAL_CHAT_MODEL,
            "whisper_model": getattr(config, 'WHISPER_MODEL', 'whisper-1')
        }
        
        point_ids = await self.qdrant_service.save_summary(
            file_path=file_path,
            summary=summary,
            file_type="audio",
            metadata=metadata
        )
        
        print(f"✅ Аудио выжимка создана и сохранена")
        
        return SummaryResponse(
            id=point_ids[0] if len(point_ids) == 1 else f"session_{point_ids[0][:8]}",
            summary=summary,
            file_name=file_name,
            file_type="audio",
            chunks_count=len(point_ids),
            is_chunked=len(point_ids) > 1,
            created_at="2025-06-30T22:00:00"
        )
    
    async def create_video_summary(
        self,
        video_path: str,
        file_name: str,
        language: str = "ru"
    ) -> SummaryResponse:
        """Создает выжимку из видеофайла."""
        print(f"🎬 Обработка видеофайла: {file_name}")
        
        # Проверяем существующую выжимку
        existing = await self.qdrant_service.check_file_exists(video_path)
        if existing:
            print(f"✅ Найдена существующая выжимка для файла {file_name}")
            return SummaryResponse(
                id=existing["id"],
                summary=existing["summary"],
                file_name=file_name,
                file_type="video",
                chunks_count=existing.get("chunks_count", 1),
                is_chunked=existing.get("is_chunked", False),
                created_at=existing["created_at"]
            )
        
        # Извлекаем аудио из видео
        print("🎬 Извлекаем аудио из видео...")
        audio_path = config.TMP_AUDIO
        extract_audio(video_path, audio_path)
        
        # Транскрибируем аудио
        print("🗣️ Транскрибируем аудио...")
        transcript = transcribe_audio(audio_path, language)
        print(f"📝 Транскрипция получена (длина: {len(transcript)} символов)")
        
        # Создаем выжимку
        summary = await self._create_summary_with_local_model(transcript)
        
        # Сохраняем в Qdrant
        metadata = {
            "language": language,
            "transcript_length": len(transcript),
            "model": config.LOCAL_CHAT_MODEL,
            "whisper_model": getattr(config, 'WHISPER_MODEL', 'whisper-1')
        }
        
        point_ids = await self.qdrant_service.save_summary(
            file_path=video_path,
            summary=summary,
            file_type="video",
            metadata=metadata
        )
        
        print(f"✅ Видео выжимка создана и сохранена")
        
        return SummaryResponse(
            id=point_ids[0] if len(point_ids) == 1 else f"session_{point_ids[0][:8]}",
            summary=summary,
            file_name=file_name,
            file_type="video",
            chunks_count=len(point_ids),
            is_chunked=len(point_ids) > 1,
            created_at="2025-06-30T22:00:00"
        )
    
    async def search_summaries(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3
    ) -> List[SearchResult]:
        """Поиск выжимок по запросу."""
        print(f"🔍 Поиск по запросу: '{query}' (мин. схожесть: {min_score})")
        
        results = await self.qdrant_service.search_similar_summaries(
            query=query,
            limit=limit,
            min_score=min_score
        )
        
        search_results = [
            SearchResult(
                id=result["id"],
                score=result["score"],
                file_name=result["file_name"],
                file_type=result["file_type"],
                created_at=result["created_at"][:10],
                summary=result["summary"][:300] + "..." if len(result["summary"]) > 300 else result["summary"],
                chunks_count=result.get("chunks_count", 1),
                is_chunked=result.get("is_chunked", False)
            )
            for result in results
        ]
        
        print(f"✅ Найдено {len(search_results)} результатов")
        return search_results
    
    async def get_all_summaries(self) -> List[SummaryResponse]:
        """Получает все сохраненные выжимки."""
        print("📚 Получение списка всех выжимок...")
        
        # Получаем все точки из коллекции
        results = await self.qdrant_service.client.scroll(
            collection_name=self.qdrant_service.collection_name,
            limit=1000,
            with_payload=True
        )
        
        points = results[0]
        
        if not points:
            return []
        
        # Группируем точки по session_id для объединения чанков
        session_groups = {}
        
        for point in points:
            payload = point.payload
            session_id = payload.get('session_id', point.id)
            
            if session_id not in session_groups:
                session_groups[session_id] = {
                    'file_name': payload.get('file_name', 'Неизвестно'),
                    'file_type': payload.get('file_type', 'unknown'),
                    'created_at': payload.get('created_at', ''),
                    'summary_length': payload.get('summary_length', 0),
                    'chunks': [],
                    'full_summary': payload.get('full_summary')
                }
            
            session_groups[session_id]['chunks'].append({
                'id': point.id,
                'chunk_text': payload.get('chunk_text', ''),
                'chunk_index': payload.get('chunk_index', 0),
                'is_chunk': payload.get('is_chunk', False)
            })
        
        # Формируем ответ
        summaries = []
        for session_id, group in session_groups.items():
            # Сортируем чанки по индексу
            group['chunks'].sort(key=lambda x: x['chunk_index'])
            
            # Восстанавливаем полный текст
            if group['full_summary']:
                summary_text = group['full_summary']
            else:
                chunk_texts = [chunk['chunk_text'] for chunk in group['chunks']]
                summary_text = self.qdrant_service._reconstruct_from_chunks(chunk_texts)
            
            summaries.append(SummaryResponse(
                id=session_id,
                summary=summary_text,
                file_name=group['file_name'],
                file_type=group['file_type'],
                chunks_count=len(group['chunks']),
                is_chunked=any(chunk['is_chunk'] for chunk in group['chunks']),
                created_at=group['created_at'][:10] if group['created_at'] else "2025-06-30"
            ))
        
        # Сортируем по дате создания (новые сначала)
        summaries.sort(key=lambda x: x.created_at, reverse=True)
        
        print(f"✅ Найдено {len(summaries)} выжимок")
        return summaries
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Удаляет временный файл."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Удален временный файл: {file_path}")
        except Exception as e:
            print(f"⚠️ Ошибка при удалении временного файла {file_path}: {e}")


# Создаем глобальный экземпляр сервиса
summary_service = SummaryService() 