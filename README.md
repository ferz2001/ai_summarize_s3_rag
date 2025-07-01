# AI Summary Service

Сервис для создания выжимок из аудио, видео и текста с сохранением в векторной базе данных Qdrant.

## Возможности

- 🎵 **Обработка аудио** - транскрипция и создание выжимок из аудиофайлов
- 🎬 **Обработка видео** - извлечение аудио, транскрипция и создание выжимок
- 📝 **Обработка текста** - создание выжимок из произвольного текста
- 🔍 **Семантический поиск** - поиск выжимок по смыслу с настраиваемым порогом
- 🧩 **Чанкование** - разбиение длинных текстов на фрагменты для лучшего поиска
- 🤖 **Локальные модели** - поддержка Docker Models с fallback на OpenAI
- 🐳 **Docker** - полная контейнеризация сервиса

## Архитектура

- **FastAPI** - веб-сервер с REST API
- **Qdrant** - векторная база данных для хранения эмбеддингов
- **Docker Models** - локальные языковые модели для генерации выжимок
- **OpenAI** - fallback для генерации и эмбеддингов
- **Whisper** - транскрипция аудио через OpenAI API
- **FFmpeg** - обработка видео/аудио файлов

## Быстрый запуск

### Через Docker Compose (рекомендуется)

1. Склонируйте репозиторий:
```bash
git clone <repository-url>
cd ai
```

2. Скопируйте и настройте переменные окружения:
```bash
cp .env.example .env
# Отредактируйте .env файл, добавив ваш OPENAI_API_KEY
```

3. Запустите сервисы:
```bash
docker-compose up -d
```

4. Сервис будет доступен по адресу: http://localhost:8000

### Локальная разработка

1. Установите зависимости:
```bash
uv sync
```

2. Запустите Qdrant:
```bash
docker-compose up qdrant -d
```

3. Запустите сервер:
```bash
cd src && python main.py
```

## API Эндпоинты

### Информация о сервисе
```
GET /
```

### Создание выжимок
```
POST /summarize/text
POST /summarize/audio
POST /summarize/video
```

### Поиск и управление
```
POST /search
GET /summaries
GET /health
```

### Примеры использования

#### Создание выжимки из текста
```bash
curl -X POST "http://localhost:8000/summarize/text" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Ваш длинный текст...",
       "file_path": "example.txt"
     }'
```

#### Загрузка аудиофайла
```bash
curl -X POST "http://localhost:8000/summarize/audio" \
     -F "file=@audio.ogg" \
     -F "language=ru"
```

#### Поиск выжимок
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "ваш поисковый запрос",
       "limit": 5,
       "min_score": 0.3
     }'
```

## Конфигурация

Основные переменные окружения:

```env
# OpenAI API
OPENAI_API_KEY=your_api_key_here

# Локальные модели
LOCAL_MODELS_URL=http://model-runner.docker.internal/engines/llama.cpp/v1
LOCAL_CHAT_MODEL=ai/smollm2
LOCAL_EMBEDDING_MODEL=ai/mxbai-embed-large
USE_LOCAL_MODELS=true

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=summaries
```

## Мониторинг

- Проверка здоровья: `GET /health`
- Логи Docker: `docker-compose logs ai-summary`
- Интерфейс Qdrant: http://localhost:6333/dashboard

## Разработка

Структура проекта:
```
├── src/
│   ├── core/
│   │   └── config.py          # Конфигурация
│   ├── main.py                # FastAPI сервер
│   ├── qdrant_manager.py      # Работа с Qdrant
│   ├── local_models_client.py # Клиент локальных моделей
│   └── tools.py               # Утилиты обработки медиа
├── Dockerfile                 # Сборка контейнера
├── docker-compose.yml         # Orchestration
└── pyproject.toml            # Зависимости
```

## Лицензия

MIT License
