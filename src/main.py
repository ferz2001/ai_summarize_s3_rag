from fastapi import FastAPI
import uvicorn

from api import api_router


# Создание FastAPI приложения
app = FastAPI(
    title="AI Summary Service",
    description="Минималистичный сервис суммаризации через LangChain",
    version="2.0.0"
)

# Подключаем API роуты
app.include_router(api_router)


if __name__ == "__main__":
    print("🚀 Запуск AI Summary Service v2.0 (LangChain)...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
