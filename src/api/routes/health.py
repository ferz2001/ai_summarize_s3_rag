from fastapi import APIRouter
import httpx

from schemas.common import HealthStatus
from core.config import config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Проверка здоровья системы."""
    
    # Проверяем Qdrant
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{config.QDRANT_HOST}:{config.QDRANT_PORT}/")
            qdrant_status = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        qdrant_status = "unhealthy"
    
    # Проверяем локальные модели если используются
    local_models_status = None
    if config.USE_LOCAL_MODELS:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{config.LOCAL_MODELS_URL}/health")
                local_models_status = "healthy" if response.status_code == 200 else "unhealthy"
        except Exception:
            local_models_status = "unhealthy"
    
    # Общий статус
    overall_status = "healthy" if qdrant_status == "healthy" and (
        not config.USE_LOCAL_MODELS or local_models_status == "healthy"
    ) else "unhealthy"
    
    return HealthStatus(
        status=overall_status,
        qdrant_status=qdrant_status,
        local_models_status=local_models_status,
        message="Система работает через LangChain" if overall_status == "healthy" else "Есть проблемы"
    ) 