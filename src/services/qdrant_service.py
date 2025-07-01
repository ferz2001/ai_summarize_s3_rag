import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import re
import httpx

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantService:
    """Сервис для работы с Qdrant векторной базой данных."""
    
    def __init__(self, host: str = None, port: int = None, collection_name: str = None):
        from core.config import config
        
        self.client = AsyncQdrantClient(
            host=host or config.QDRANT_HOST,
            port=port or config.QDRANT_PORT
        )
        self.collection_name = collection_name or config.QDRANT_COLLECTION_NAME
    
    async def _ensure_collection_exists(self):
        """Создает коллекцию, если она не существует."""
        from core.config import config
        
        try:
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            # Определяем размерность векторов в зависимости от используемой модели
            vector_size = 1024 if config.USE_LOCAL_MODELS else 1536
            
            if self.collection_name not in collection_names:
                print(f"🔧 Создаю коллекцию '{self.collection_name}' в Qdrant (размерность: {vector_size})...")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print("✅ Коллекция создана успешно")
            else:
                # Проверяем размерность существующей коллекции
                collection_info = await self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != vector_size:
                    print(f"⚠️ Размерность коллекции не соответствует модели ({existing_size} != {vector_size})")
                    print(f"🔄 Пересоздаю коллекцию '{self.collection_name}' с правильной размерностью...")
                    
                    # Удаляем старую коллекцию
                    await self.client.delete_collection(self.collection_name)
                    
                    # Создаём новую с правильной размерностью
                    await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                    print("✅ Коллекция пересоздана с правильной размерностью")
                else:
                    print(f"✅ Коллекция '{self.collection_name}' уже существует (размерность: {vector_size})")
        except Exception as e:
            print(f"❌ Ошибка при работе с коллекцией: {e}")
            raise
    
    async def _get_text_embedding(self, text: str) -> List[float]:
        """Получает эмбеддинг текста через локальные модели или OpenAI."""
        from core.config import config
        
        if config.USE_LOCAL_MODELS:
            # Используем только локальные модели, без fallback
            from services.local_models_service import LocalModelsService
            
            local_models = LocalModelsService()
            embeddings = await local_models.get_embeddings([text])
            return embeddings[0]
        else:
            # Используем OpenAI если локальные модели отключены
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": text,
                        "model": "text-embedding-3-small"
                    }
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Генерирует хеш файла для уникальной идентификации."""
        file_path_obj = Path(file_path)
        
        # Если файл существует, используем его метаданные
        if file_path_obj.exists():
            stat = file_path_obj.stat()
            hash_string = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        else:
            # Для виртуальных файлов (например, manual_input) используем просто путь
            hash_string = f"{file_path}_virtual"
        
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Разбивает текст на чанки с перекрытием.
        
        Args:
            text: Исходный текст
            max_chunk_size: Максимальный размер чанка в символах
            overlap: Размер перекрытия между чанками
            
        Returns:
            Список чанков
        """
        # Разбиваем по предложениям для лучшего контекста
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если добавление предложения не превысит лимит
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Начинаем новый чанк
                current_chunk = sentence
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Добавляем перекрытие между чанками
        if len(chunks) > 1 and overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped_chunks.append(chunk)
                else:
                    # Берем последние overlap символов из предыдущего чанка
                    prev_chunk = chunks[i-1]
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    overlapped_chunk = overlap_text + "... " + chunk
                    overlapped_chunks.append(overlapped_chunk)
            return overlapped_chunks
        
        return chunks
    
    async def save_summary(
        self, 
        file_path: str, 
        summary: str, 
        file_type: str = "audio",
        metadata: Optional[Dict] = None,
        use_chunks: bool = True
    ) -> List[str]:
        """
        Сохраняет выжимку в Qdrant.
        
        Args:
            file_path: Путь к исходному файлу
            summary: Текст выжимки
            file_type: Тип файла (audio, video, text)
            metadata: Дополнительные метаданные
            use_chunks: Разбивать ли текст на чанки
            
        Returns:
            Список ID сохраненных записей
        """
        try:
            # Убеждаемся что коллекция существует
            await self._ensure_collection_exists()
            
            # Создаем базовые метаданные
            file_hash = self._generate_file_hash(file_path)
            session_id = str(uuid.uuid4())  # Общий ID для всех чанков одного файла
            
            base_payload = {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_hash": file_hash,
                "file_type": file_type,
                "created_at": datetime.now().isoformat(),
                "summary_length": len(summary),
                "session_id": session_id,  # Связь между чанками
                **(metadata or {})
            }
            
            # Разбиваем на чанки или сохраняем целиком
            if use_chunks and len(summary) > 300:  # Разбиваем только длинные тексты
                chunks = self._split_text_into_chunks(summary)
                print(f"🔄 Разбиваю выжимку на {len(chunks)} чанков...")
            else:
                chunks = [summary]
                print("🔄 Сохраняю выжимку целиком...")
            
            points_to_save = []
            point_ids = []
            
            for i, chunk in enumerate(chunks):
                # Создаем эмбеддинг для каждого чанка
                print(f"🔄 Создаю эмбеддинг для чанка {i+1}/{len(chunks)}...")
                embedding = await self._get_text_embedding(chunk)
                
                point_id = str(uuid.uuid4())
                point_ids.append(point_id)
                
                payload = {
                    **base_payload,
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_chunk": len(chunks) > 1,
                    "full_summary": summary if len(chunks) == 1 else None  # Полный текст только для нечанкованных
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points_to_save.append(point)
            
            # Сохраняем все чанки одним запросом
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points_to_save
            )
            
            if len(chunks) > 1:
                print(f"✅ Выжимка сохранена как {len(chunks)} чанков в Qdrant (session: {session_id})")
            else:
                print(f"✅ Выжимка сохранена в Qdrant с ID: {point_ids[0]}")
            
            return point_ids
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении в Qdrant: {e}")
            raise
    
    async def search_similar_summaries(self, query: str, limit: int = 10, min_score: float = 0.3) -> List[Dict]:
        """
        Ищет похожие выжимки по запросу.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов (отдельных чанков)
            min_score: Минимальный порог схожести (0.0-1.0). По умолчанию 0.3
            
        Returns:
            Список найденных чанков с метаданными, отсортированных по релевантности
        """
        try:
            query_embedding = await self._get_text_embedding(query)
            
            # Ищем больше результатов, чтобы учесть чанки
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 3,  # Ищем в 3 раза больше, чтобы учесть чанки
                with_payload=True
            )
            
            # Фильтруем результаты по минимальному порогу схожести
            filtered_results = [result for result in results if result.score >= min_score]
            
            if not filtered_results:
                return []
            
            # Группируем результаты по session_id (файлам)
            file_groups = {}
            
            for result in filtered_results:
                payload = result.payload
                session_id = payload.get('session_id', result.id)
                
                if session_id not in file_groups:
                    # Создаем новую группу для файла
                    file_groups[session_id] = {
                        "id": session_id,
                        "file_name": payload.get('file_name', 'Неизвестно'),
                        "file_type": payload.get('file_type', 'unknown'),
                        "file_path": payload.get('file_path', ''),
                        "created_at": payload.get('created_at', ''),
                        "summary_length": payload.get('summary_length', 0),
                        "best_score": result.score,
                        "chunks": [],
                        "full_summary": payload.get('full_summary')  # Для нечанковантекстовных 
                    }
                
                # Добавляем чанк к группе
                chunk_info = {
                    "chunk_id": result.id,
                    "score": result.score,
                    "chunk_text": payload.get('chunk_text', ''),
                    "chunk_index": payload.get('chunk_index', 0),
                    "is_chunk": payload.get('is_chunk', False)
                }
                file_groups[session_id]["chunks"].append(chunk_info)
                
                # Обновляем лучший скор
                if result.score > file_groups[session_id]["best_score"]:
                    file_groups[session_id]["best_score"] = result.score
            
            # Собираем все чанки из всех групп и сортируем по скору
            all_chunks = []
            for session_id, group in file_groups.items():
                for chunk in group["chunks"]:
                    chunk_result = {
                        "id": chunk["chunk_id"],
                        "score": chunk["score"],
                        "file_name": group["file_name"],
                        "file_type": group["file_type"],
                        "file_path": group["file_path"],
                        "created_at": group["created_at"],
                        "summary_length": group["summary_length"],
                        "summary": chunk["chunk_text"][:300] + "..." if len(chunk["chunk_text"]) > 300 else chunk["chunk_text"],
                        "chunk_index": chunk["chunk_index"],
                        "is_chunk": chunk["is_chunk"],
                        "session_id": session_id,
                        "total_chunks": group.get("total_chunks", 1)
                    }
                    all_chunks.append(chunk_result)
            
            # Сортируем все чанки по скору и берем топ результатов
            sorted_chunks = sorted(all_chunks, key=lambda x: x["score"], reverse=True)
            final_results = sorted_chunks[:limit]
            
            return final_results
            
        except Exception as e:
            print(f"❌ Ошибка при поиске: {e}")
            raise
    
    async def check_file_exists(self, file_path: str) -> Optional[Dict]:
        """
        Проверяет, есть ли уже выжимка для данного файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Информация о существующей выжимке или None
        """
        try:
            file_hash = self._generate_file_hash(file_path)
            
            results = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_hash",
                            "match": {"value": file_hash}
                        }
                    ]
                },
                limit=100,  # Больше лимит для получения всех чанков
                with_payload=True
            )
            
            if results[0]:  # results = (points, next_page_offset)
                points = results[0]
                
                # Если есть несколько чанков, объединяем их
                if len(points) > 1:
                    # Сортируем чанки по индексу
                    sorted_points = sorted(points, key=lambda p: p.payload.get('chunk_index', 0))
                    
                    # Собираем полный текст из чанков
                    chunk_texts = []
                    session_id = sorted_points[0].payload.get('session_id')
                    
                    for point in sorted_points:
                        chunk_text = point.payload.get('chunk_text', '')
                        if chunk_text:
                            chunk_texts.append(chunk_text)
                    
                    # Объединяем чанки, убирая перекрытия
                    full_summary = self._reconstruct_from_chunks(chunk_texts)
                    
                    return {
                        "id": session_id,
                        "summary": full_summary,
                        "chunks_count": len(points),
                        "is_chunked": True,
                        **sorted_points[0].payload
                    }
                else:
                    # Один чанк или нечанкованный текст
                    point = points[0]
                    summary = point.payload.get('full_summary') or point.payload.get('chunk_text', '')
                    
                    return {
                        "id": point.id,
                        "summary": summary,
                        "chunks_count": 1,
                        "is_chunked": False,
                        **point.payload
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Ошибка при проверке файла: {e}")
            return None
    
    def _reconstruct_from_chunks(self, chunk_texts: List[str]) -> str:
        """
        Восстанавливает полный текст из чанков, убирая дублированные части.
        
        Args:
            chunk_texts: Список текстов чанков
            
        Returns:
            Восстановленный полный текст
        """
        if not chunk_texts:
            return ""
        
        if len(chunk_texts) == 1:
            return chunk_texts[0]
        
        result = chunk_texts[0]
        
        for i in range(1, len(chunk_texts)):
            current_chunk = chunk_texts[i]
            
            # Убираем начальное перекрытие (ищем общую часть)
            if "..." in current_chunk:
                # Если есть разделитель, берем часть после него
                parts = current_chunk.split("...", 1)
                if len(parts) > 1:
                    new_part = parts[1].strip()
                    result += ". " + new_part
                else:
                    result += ". " + current_chunk
            else:
                result += ". " + current_chunk
        
        return result 