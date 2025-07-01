import asyncio
from typing import List, Optional
import httpx
from core.config import config


class LocalModelsService:
    """Сервис для работы с локальными моделями Docker Models."""
    
    def __init__(self):
        self.base_url = config.LOCAL_MODELS_URL
        self.chat_model = config.LOCAL_CHAT_MODEL
        self.embedding_model = config.LOCAL_EMBEDDING_MODEL
        self.use_local = config.USE_LOCAL_MODELS
        
    async def chat_completion(
        self, 
        messages: List[dict], 
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Отправляет запрос на генерацию текста.
        
        Args:
            messages: Список сообщений в формате OpenAI
            model: Модель для использования (по умолчанию из конфига)
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            
        Returns:
            Сгенерированный текст
        """
        if not self.use_local:
            raise RuntimeError("Локальные модели отключены в конфигурации")
            
        model_name = model or self.chat_model
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Неожиданный формат ответа от модели")
                    
            except httpx.TimeoutException:
                raise RuntimeError("Превышено время ожидания ответа от локальной модели")
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ошибка HTTP от локальной модели: {e.response.status_code}")
            except Exception as e:
                raise RuntimeError(f"Ошибка при запросе к локальной модели: {e}")
    
    async def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Получает эмбеддинги для списка текстов.
        
        Args:
            texts: Список текстов для получения эмбеддингов
            model: Модель для эмбеддингов (по умолчанию из конфига)
            
        Returns:
            Список эмбеддингов
        """
        if not self.use_local:
            raise RuntimeError("Локальные модели отключены в конфигурации")
            
        model_name = model or self.embedding_model
        
        url = f"{self.base_url}/embeddings"
        
        payload = {
            "model": model_name,
            "input": texts
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                if "data" in data:
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise ValueError("Неожиданный формат ответа от модели эмбеддингов")
                    
            except httpx.TimeoutException:
                raise RuntimeError("Превышено время ожидания ответа от модели эмбеддингов")
            except httpx.HTTPStatusError as e:
                # Не используем fallback на OpenAI, выбрасываем ошибку
                raise RuntimeError(f"Локальная модель эмбеддингов недоступна: {e.response.status_code}")
            except Exception as e:
                raise RuntimeError(f"Ошибка локальной модели эмбеддингов: {e}")
