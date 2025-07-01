from fastapi import APIRouter

from api.routes import summary, health, chat, add

# Создаем главный роутер для API
api_router = APIRouter()

# Подключаем все роуты
api_router.include_router(summary.router)
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(add.router)
