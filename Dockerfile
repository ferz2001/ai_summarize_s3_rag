FROM python:3.12-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем uv для управления зависимостями
RUN pip install uv

# Устанавливаем зависимости
RUN uv sync --frozen

# Создаем директории для файлов
RUN mkdir -p /app/uploads/audio /app/uploads/video /app/uploads/temp

# Открываем порт
EXPOSE 8000

# Переменные окружения
ENV PYTHONPATH=/app/src
ENV TMP_AUDIO=/tmp/extracted_audio.wav

# Команда запуска с hot reload
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 