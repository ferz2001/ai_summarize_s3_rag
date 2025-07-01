from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str = ""

    # Файлы и хранилище
    UPLOADS_DIR: str = "/app/uploads"
    TEMP_DIR: str = "/app/uploads/temp"
    TMP_AUDIO: str = "/app/uploads/temp/extracted_audio.wav"
    
    # Локальные модели Docker Models
    LOCAL_MODELS_URL: str = "http://model-runner.docker.internal/engines/llama.cpp/v1"
    LOCAL_CHAT_MODEL: str = "ai/smollm2"
    LOCAL_WHISPER_MODEL: str = "small"
    LOCAL_EMBEDDING_MODEL: str = "ai/mxbai-embed-large"
    USE_LOCAL_MODELS: bool = True
    
    # Qdrant настройки
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "summaries"


config = Config()
