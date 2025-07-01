from fastapi import APIRouter

from api.routes import summary, search, health

# Создаем главный роутер для API
api_router = APIRouter()

# Подключаем все роуты
api_router.include_router(summary.router)
api_router.include_router(search.router) 
api_router.include_router(health.router)
