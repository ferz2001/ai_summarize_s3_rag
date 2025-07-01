import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

from core.config import config


class FileService:
    """Сервис для управления файлами."""
    
    def __init__(self):
        self.uploads_dir = Path(config.UPLOADS_DIR)
        self.temp_dir = Path(config.TEMP_DIR)
        self.audio_dir = self.uploads_dir / "audio"
        self.video_dir = self.uploads_dir / "video"
        
        # Создаем директории если их нет
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Создает необходимые директории."""
        for directory in [self.uploads_dir, self.temp_dir, self.audio_dir, self.video_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"📁 Директория готова: {directory}")
    
    async def save_uploaded_file(
        self, 
        file: UploadFile, 
        file_type: str = "temp",
        preserve_extension: bool = True
    ) -> str:
        """
        Сохраняет загруженный файл.
        
        Args:
            file: Загруженный файл
            file_type: Тип файла (audio, video, temp)
            preserve_extension: Сохранять ли расширение файла
            
        Returns:
            Путь к сохраненному файлу
        """
        # Определяем директорию
        if file_type == "audio":
            target_dir = self.audio_dir
        elif file_type == "video":
            target_dir = self.video_dir
        else:
            target_dir = self.temp_dir
        
        # Генерируем уникальное имя файла
        file_id = str(uuid.uuid4())
        
        if preserve_extension and file.filename:
            # Извлекаем расширение
            extension = Path(file.filename).suffix
            filename = f"{file_id}{extension}"
        else:
            filename = file_id
        
        file_path = target_dir / filename
        
        # Сохраняем файл
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            print(f"💾 Файл сохранен: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")
            raise
    
    def copy_to_temp(self, source_path: str, filename: Optional[str] = None) -> str:
        """
        Копирует файл во временную директорию.
        
        Args:
            source_path: Путь к исходному файлу
            filename: Имя файла (если не указано, используется UUID)
            
        Returns:
            Путь к временному файлу
        """
        if not filename:
            source_ext = Path(source_path).suffix
            filename = f"{uuid.uuid4()}{source_ext}"
        
        temp_path = self.temp_dir / filename
        shutil.copy2(source_path, temp_path)
        
        print(f"📋 Файл скопирован во временную папку: {temp_path}")
        return str(temp_path)
    
    def cleanup_file(self, file_path: str):
        """
        Удаляет файл.
        
        Args:
            file_path: Путь к файлу для удаления
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Файл удален: {file_path}")
        except Exception as e:
            print(f"⚠️ Ошибка при удалении файла {file_path}: {e}")
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Получает информацию о файле.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Словарь с информацией о файле
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        
        return {
            "exists": True,
            "name": path.name,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "extension": path.suffix,
            "type": self._get_file_type_from_path(str(path))
        }
    
    def _get_file_type_from_path(self, file_path: str) -> str:
        """Определяет тип файла по пути."""
        path = Path(file_path)
        
        if self.audio_dir in path.parents:
            return "audio"
        elif self.video_dir in path.parents:
            return "video"
        elif self.temp_dir in path.parents:
            return "temp"
        else:
            return "unknown"
    
    def list_files(self, file_type: Optional[str] = None) -> list:
        """
        Возвращает список файлов.
        
        Args:
            file_type: Тип файлов (audio, video, temp, None для всех)
            
        Returns:
            Список путей к файлам
        """
        files = []
        
        if file_type == "audio" or file_type is None:
            files.extend([str(f) for f in self.audio_dir.glob("*") if f.is_file()])
        
        if file_type == "video" or file_type is None:
            files.extend([str(f) for f in self.video_dir.glob("*") if f.is_file()])
        
        if file_type == "temp" or file_type is None:
            files.extend([str(f) for f in self.temp_dir.glob("*") if f.is_file()])
        
        return sorted(files)
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Очищает старые временные файлы.
        
        Args:
            max_age_hours: Максимальный возраст файлов в часах
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        
        for temp_file in self.temp_dir.glob("*"):
            if temp_file.is_file():
                file_age = current_time - temp_file.stat().st_mtime
                
                if file_age > max_age_seconds:
                    try:
                        temp_file.unlink()
                        deleted_count += 1
                        print(f"🗑️ Удален старый временный файл: {temp_file}")
                    except Exception as e:
                        print(f"⚠️ Не удалось удалить файл {temp_file}: {e}")
        
        if deleted_count > 0:
            print(f"🧹 Очищено {deleted_count} временных файлов")


# Глобальный экземпляр сервиса
file_service = FileService() 