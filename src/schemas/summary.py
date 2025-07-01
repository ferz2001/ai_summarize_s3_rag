from pydantic import BaseModel, Field
from typing import List


class SummaryResponse(BaseModel):
    """Ответ с информацией о выжимке."""
    id: str = Field(..., description="ID выжимки")
    summary: str = Field(..., description="Текст выжимки")
    file_name: str = Field(..., description="Имя файла")
    file_type: str = Field(..., description="Тип файла")
    chunks_count: int = Field(..., description="Количество чанков")
    is_chunked: bool = Field(..., description="Разбита ли на чанки")
    created_at: str = Field(..., description="Дата создания")


class TextSummaryRequest(BaseModel):
    """Запрос на создание выжимки из текста."""
    text: str = Field(..., description="Текст для создания выжимки")
    file_path: str = Field(default="manual_input", description="Путь к файлу или описание источника")


class ListSummariesResponse(BaseModel):
    """Ответ со списком всех выжимок."""
    summaries: List[SummaryResponse] = Field(..., description="Список всех выжимок")
    total_files: int = Field(..., description="Общее количество файлов")
    total_records: int = Field(..., description="Общее количество записей в БД") 