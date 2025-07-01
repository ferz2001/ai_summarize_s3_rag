from pydantic import BaseModel, Field
from typing import List


class SearchRequest(BaseModel):
    """Запрос на поиск выжимок."""
    query: str = Field(..., description="Поисковый запрос")
    limit: int = Field(default=5, ge=1, le=50, description="Максимальное количество результатов")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Минимальный порог схожести")


class SearchResult(BaseModel):
    """Результат поиска выжимки."""
    id: str = Field(..., description="ID результата")
    score: float = Field(..., description="Скор схожести")
    file_name: str = Field(..., description="Имя файла")
    file_type: str = Field(..., description="Тип файла")
    created_at: str = Field(..., description="Дата создания")
    summary: str = Field(..., description="Превью выжимки")
    chunks_count: int = Field(..., description="Количество чанков")
    is_chunked: bool = Field(..., description="Разбита ли на чанки")


class SearchResponse(BaseModel):
    """Ответ с результатами поиска."""
    results: List[SearchResult] = Field(..., description="Результаты поиска")
    total: int = Field(..., description="Общее количество найденных результатов")
    query: str = Field(..., description="Поисковый запрос")
    min_score: float = Field(..., description="Использованный порог схожести") 