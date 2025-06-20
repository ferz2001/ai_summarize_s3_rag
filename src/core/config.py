from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    OLLAMA_MODEL: str
    WHISPER_MODEL: str
    TMP_AUDIO: str
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")


config = Config()
