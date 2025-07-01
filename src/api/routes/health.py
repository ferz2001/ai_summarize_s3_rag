from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas.health import HealthResponse
from services.qdrant_service import QdrantService
from services.local_models_service import LocalModelsService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse, summary="Проверка здоровья сервиса")
async def health_check():
    """Проверка состояния сервиса и подключений."""
    try:
        status = {
            "status": "healthy",
            "qdrant": "unknown",
            "local_models": "unknown"
        }
        
        # Проверяем Qdrant
        try:
            qdrant_service = QdrantService()
            await qdrant_service.client.get_collections()
            status["qdrant"] = "connected"
        except Exception:
            status["qdrant"] = "disconnected"
        
        # Проверяем локальные модели
        try:
            local_models = LocalModelsService()
            test_response = await local_models.chat_completion([
                {"role": "user", "content": "Привет, это тест"}
            ])
            if test_response:
                status["local_models"] = "connected"
        except Exception:
            status["local_models"] = "disconnected"
        
        return HealthResponse(**status)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy", 
                "qdrant": "unknown",
                "local_models": "unknown",
                "error": str(e)
            }
        ) 