from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from api import api_router
from services.qdrant_service import QdrantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    print("🚀 Запуск AI Summary Service...")
    try:
        qdrant_service = QdrantService()
        await qdrant_service._ensure_collection_exists()
        print("✅ Подключение к Qdrant установлено")
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Завершение работы сервера...")


# Создание FastAPI приложения
app = FastAPI(
    title="AI Summary Service",
    description="Сервис для создания выжимок из аудио, видео и текста с сохранением в Qdrant",
    version="1.0.0",
    lifespan=lifespan
)

# Подключаем API роуты
app.include_router(api_router)


if __name__ == "__main__":
    print("🚀 Запуск AI Summary Service...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
