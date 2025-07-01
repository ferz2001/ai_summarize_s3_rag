from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса."""
    status: str = Field(..., description="Общий статус сервиса")
    qdrant: str = Field(..., description="Статус подключения к Qdrant")
    local_models: str = Field(..., description="Статус подключения к локальным моделям")
    error: Optional[str] = Field(None, description="Описание ошибки, если есть") 