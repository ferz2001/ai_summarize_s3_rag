import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid
import hashlib
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config import config
from services.local_models_service import LocalModelsService


class LocalModelsEmbeddings(Embeddings):
    """Кастомный класс эмбеддингов для работы с Local Models API."""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Создает эмбеддинги для списка документов."""
        import requests
        
        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                return [item["embedding"] for item in data["data"]]
            else:
                raise ValueError("Неожиданный формат ответа от модели эмбеддингов")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ошибка локальной модели эмбеддингов: {e}")
    
    def embed_query(self, text: str) -> List[float]:
        """Создает эмбеддинг для одного запроса."""
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Асинхронное создание эмбеддингов для списка документов."""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Асинхронное создание эмбеддинга для одного запроса."""
        return self.embed_query(text)


class RAGService:
    """Сервис для поиска документов и генерации ответов через RAG."""
    
    def __init__(self):
        self.async_client = AsyncQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        self.collection_name = f"{config.QDRANT_COLLECTION_NAME}"
        
        # Инициализируем эмбеддинги
        if config.USE_LOCAL_MODELS:
            self.local_models = LocalModelsService()
            self.embeddings_model = LocalModelsEmbeddings(
                base_url=config.LOCAL_MODELS_URL,
                model=config.LOCAL_EMBEDDING_MODEL
            )
            self.embedding_dim = 1024
            print(f"🤖 Использую локальную модель эмбеддингов: {config.LOCAL_EMBEDDING_MODEL}")
        else:
            self.embeddings_model = OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            self.embedding_dim = 1536
            print(f"🌐 Использую OpenAI эмбеддинги: text-embedding-3-small")
        
        # Инициализируем сплиттер текста для разбивки документов
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=False
        )
        
        # Инициализируем векторное хранилище
        self.vector_store = self._init_vector_store()
    
    def _init_vector_store(self) -> QdrantVectorStore:
        """Инициализирует векторное хранилище LangChain."""
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        
        try:
            collection_info = client.get_collection(self.collection_name)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != self.embedding_dim:
                print(f"🔄 Пересоздаю коллекцию '{self.collection_name}' - неправильная размерность ({existing_size} -> {self.embedding_dim})")
                client.delete_collection(self.collection_name)
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Коллекция '{self.collection_name}' пересоздана")
            else:
                print(f"✅ Коллекция '{self.collection_name}' существует")
        except Exception:
            print(f"🔄 Создаю коллекцию '{self.collection_name}'...")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Коллекция '{self.collection_name}' создана")
        
        return QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings_model
        )
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Генерирует хеш файла для проверки дубликатов."""
        file_path = Path(file_path)
        
        if file_path.exists():
            stat = file_path.stat()
            hash_string = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
        else:
            hash_string = str(file_path)
        
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    async def _ensure_collection_exists(self):
        """Убеждаемся, что коллекция существует."""
        try:
            await self.async_client.get_collection(self.collection_name)
        except Exception:
            print(f"🔄 Создаю коллекцию '{self.collection_name}'...")
            await self.async_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    async def check_document_exists(self, file_path: str) -> Optional[Dict]:
        """Проверяет, существует ли документ в системе."""
        try:
            file_hash = self._generate_file_hash(file_path)
            
            results = await self.async_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "file_hash",
                            "match": {"value": file_hash}
                        }
                    ]
                },
                limit=100,
                with_payload=True
            )
            
            if results[0]:
                points = results[0]
                
                if len(points) > 1:
                    # Несколько чанков - объединяем
                    sorted_points = sorted(points, key=lambda p: p.payload.get('chunk_index', 0))
                    
                    full_content = " ".join([
                        point.payload.get('page_content', '') 
                        for point in sorted_points
                    ])
                    
                    return {
                        "id": sorted_points[0].payload.get('session_id'),
                        "content": full_content,
                        "chunks_count": len(points),
                        "is_chunked": True,
                        **sorted_points[0].payload
                    }
                else:
                    # Один чанк
                    point = points[0]
                    return {
                        "id": point.id,
                        "content": point.payload.get('page_content', ''),
                        "chunks_count": 1,
                        "is_chunked": False,
                        **point.payload
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Ошибка при проверке документа: {e}")
            return None
    
    async def add_summary_to_rag(
        self,
        summary_text: str,
        original_text: str,
        file_path: str,
        file_type: str = "summary",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Добавляет выжимку в RAG систему."""
        await self._ensure_collection_exists()
        
        # Проверяем, существует ли документ
        existing = await self.check_document_exists(file_path)
        if existing:
            return {
                "status": "exists",
                "message": f"Документ уже существует в системе",
                "document_id": existing.get("id"),
                "file_path": file_path
            }
        
        # Генерируем метаданные
        file_hash = self._generate_file_hash(file_path)
        session_id = str(uuid.uuid4())
        
        base_metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "file_hash": file_hash,
            "file_type": file_type,
            "created_at": datetime.now().isoformat(),
            "content_length": len(summary_text),
            "original_length": len(original_text),
            "session_id": session_id,
            "is_summary": True,
            **(metadata or {})
        }
        
        # Разбиваем выжимку на чанки
        chunks = self.text_splitter.split_text(summary_text)
        print(f"🔄 Разбиваю выжимку на {len(chunks)} чанков...")
        
        # Создаем документы LangChain
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_chunk": len(chunks) > 1
            }
            
            doc = Document(
                page_content=chunk,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        # Добавляем документы в векторное хранилище
        ids = self.vector_store.add_documents(documents)
        
        print(f"✅ Выжимка добавлена в RAG ({len(chunks)} чанков)")
        
        return {
            "status": "added",
            "message": f"Выжимка успешно добавлена в RAG систему",
            "document_ids": ids,
            "file_path": file_path,
            "chunks_count": len(ids),
            "summary_length": len(summary_text),
            "original_length": len(original_text)
        }
    
    def search_documents(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Document]:
        """Ищет релевантные документы."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold
            }
        )
        
        documents = retriever.invoke(query)
        return documents
    
    async def generate_answer(
        self,
        question: str,
        context_docs: List[Document],
        max_context_length: int = 3000
    ) -> str:
        """Генерирует ответ на основе найденных документов."""
        # Формируем контекст из найденных документов
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            content = doc.page_content
            if current_length + len(content) <= max_context_length:
                file_name = doc.metadata.get('file_name', 'Неизвестно')
                context_parts.append(f"[Из файла: {file_name}]\n{content}")
                current_length += len(content)
            else:
                break
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Ты умный помощник, который отвечает на вопросы на основе предоставленного контекста.
            Контекст:
            {context}

            Вопрос: {question}

            Инструкции:
            - Отвечай только на основе предоставленного контекста
            - Если в контексте нет информации для ответа, скажи об этом честно
            - Отвечай кратко и по существу на русском языке
            - Если возможно, укажи из какого файла взята информация

            Ответ:
        """

        if config.USE_LOCAL_MODELS:
            response = await self.local_models.chat_completion([
                {
                    "role": "system",
                    "content": "Ты помощник для ответов на вопросы по документам. Отвечай кратко и точно на основе предоставленного контекста."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ], model=config.LOCAL_CHAT_MODEL)
            return response
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model="gpt-4o-mini",
                temperature=0.3
            )
            response = await llm.ainvoke(prompt)
            return response.content
    
    async def query(
        self,
        question: str,
        k: int = 5,
        score_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Основной метод для RAG запросов."""
        print(f"🔍 RAG поиск по запросу: '{question}'")
        
        # Ищем релевантные документы
        documents = self.search_documents(question, k=k, score_threshold=score_threshold)
        
        if not documents:
            return {
                "answer": "К сожалению, я не нашел релевантной информации для ответа на ваш вопрос.",
                "sources": [],
                "context_used": False
            }
        
        print(f"✅ Найдено {len(documents)} релевантных документов")
        
        # Генерируем ответ
        answer = await self.generate_answer(question, documents)
        
        # Формируем информацию об источниках
        sources = []
        for doc in documents:
            sources.append({
                "file_name": doc.metadata.get('file_name', 'Неизвестно'),
                "file_type": doc.metadata.get('file_type', 'unknown'),
                "chunk_index": doc.metadata.get('chunk_index', 0),
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": True,
            "documents_found": len(documents)
        }


# Глобальный экземпляр сервиса
rag_service = RAGService() 