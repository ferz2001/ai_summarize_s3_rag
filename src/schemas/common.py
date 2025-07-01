from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# === Базовые модели ===

class ChatRequest(BaseModel):
    """Запрос для чата или RAG."""
    message: str = Field(..., description="Сообщение или вопрос")
    max_results: Optional[int] = Field(5, description="Максимальное количество результатов", ge=1, le=20)
    min_score: Optional[float] = Field(0.3, description="Минимальный порог релевантности", ge=0.0, le=1.0)


class DocumentSource(BaseModel):
    """Источник документа для RAG ответа."""
    file_name: str
    file_type: str
    chunk_index: int
    content_preview: str


class RAGResponse(BaseModel):
    """Ответ RAG системы."""
    answer: str
    sources: List[DocumentSource]
    context_used: bool
    documents_found: int
    query: str


# === Суммаризация ===

class SummaryRequest(BaseModel):
    """Запрос на создание суммаризации."""
    text: Optional[str] = None
    file_path: Optional[str] = None
    speed_multiplier: float = Field(2.0, description="Множитель скорости для аудио/видео", ge=0.5, le=4.0)


class SummaryResponse(BaseModel):
    """Ответ с суммаризацией."""
    summary: str
    file_path: str
    original_length: int
    summary_length: int


# === Добавление документов в RAG ===

class AddTextRequest(BaseModel):
    """Запрос на добавление текстовой выжимки в RAG."""
    text: str = Field(..., description="Текст для создания выжимки и добавления в RAG")
    file_path: Optional[str] = Field(None, description="Путь к файлу (опционально)")
    speed_multiplier: float = Field(2.0, description="Множитель скорости для аудио/видео", ge=0.5, le=4.0)


class AddDocumentResponse(BaseModel):
    """Ответ после добавления документа в RAG."""
    success: bool
    message: str
    summary: str
    file_path: str
    summary_length: int
    original_length: int
    chunks_count: int
    document_ids: List[str]
    status: str  # "added" или "exists"


# === Здоровье системы ===

class HealthStatus(BaseModel):
    """Статус здоровья системы."""
    status: str
    qdrant_status: str
    local_models_status: Optional[str] = None
    message: str 